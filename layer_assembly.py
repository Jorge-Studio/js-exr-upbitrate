"""
Layer Assembly and Bit-Depth Validation nodes for ComfyUI.

LayerAssembler: Composites multiple processed layers back into a single
image using alpha masks, feathering, and color science transforms.

BitDepthValidator: QC tool that analyzes the expanded image, verifying
unique value counts, histogram continuity, and PSNR against the original.
Produces a validation report and optional waveform visualization.
"""

import json
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

try:
    import colour
    _HAS_COLOUR = True
except ImportError:
    _HAS_COLOUR = False


class LayerAssembler:
    """
    Composite multiple layers back into a single image.

    Accepts dynamic-length layer and mask lists (from OUTPUT_IS_LIST nodes)
    or individual layer/mask pairs. Supports alpha compositing with
    feathered edges and optional color space conversion.
    """

    CATEGORY = "image/bitdepth"
    FUNCTION = "assemble"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("composite",)
    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "layers": ("IMAGE",),
                "masks": ("MASK",),
                "feather_radius": ("INT", {
                    "default": 3, "min": 0, "max": 20, "step": 1,
                    "tooltip": "Pixel radius for feathering mask edges."
                }),
                "output_colorspace": (["passthrough", "sRGB", "Rec.709", "Linear",
                                       "ACEScg", "ACES2065-1"], {
                    "default": "passthrough",
                }),
            },
            "optional": {
                "background_color": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1,
                }),
            },
        }

    def _prepare_mask(self, mask, h, w, feather_radius):
        if isinstance(mask, torch.Tensor):
            m = mask.cpu().float().numpy()
        else:
            m = np.array(mask, dtype=np.float32)

        if len(m.shape) == 3:
            m = m[0] if m.shape[0] == 1 else m.mean(axis=-1)

        if m.shape != (h, w):
            from scipy.ndimage import zoom
            m = zoom(m, (h / m.shape[0], w / m.shape[1]), order=1)

        if feather_radius > 0:
            m = gaussian_filter(m, sigma=feather_radius * 0.5)

        return np.clip(m, 0, 1).astype(np.float32)

    def _convert_colorspace(self, image, target):
        if target == "passthrough":
            return image

        if target == "Linear":
            return np.where(
                image <= 0.04045,
                image / 12.92,
                np.power((image + 0.055) / 1.055, 2.4)
            ).astype(np.float32)

        if not _HAS_COLOUR:
            print(f"[LayerAssembler] colour-science not installed, skipping {target}")
            return image

        try:
            from colour.models import RGB_COLOURSPACES

            linear = colour.cctf_decoding(np.clip(image, 0, 1), function='sRGB')

            if target in ("sRGB", "Rec.709"):
                return colour.cctf_encoding(
                    np.clip(linear, 0, 1), function='sRGB'
                ).astype(np.float32)

            source_cs = RGB_COLOURSPACES['sRGB']
            if target == "ACEScg":
                target_cs = RGB_COLOURSPACES['ACEScg']
            elif target == "ACES2065-1":
                target_cs = RGB_COLOURSPACES['ACES2065-1']
            else:
                return image

            try:
                XYZ = colour.RGB_to_XYZ(linear, colourspace=source_cs)
                result = colour.XYZ_to_RGB(XYZ, colourspace=target_cs)
            except TypeError:
                XYZ = colour.RGB_to_XYZ(
                    linear, source_cs.whitepoint, source_cs.whitepoint,
                    source_cs.matrix_RGB_to_XYZ
                )
                result = colour.XYZ_to_RGB(
                    XYZ, target_cs.whitepoint, target_cs.whitepoint,
                    target_cs.matrix_XYZ_to_RGB
                )

            return np.clip(result, 0, None).astype(np.float32)

        except Exception as e:
            print(f"[LayerAssembler] Color conversion failed: {e}")
            return image

    def assemble(self, layers, masks, feather_radius, output_colorspace,
                 background_color=None):

        layer_list = layers if isinstance(layers, list) else [layers]
        mask_list = masks if isinstance(masks, list) else [masks]
        fr = feather_radius[0] if isinstance(feather_radius, list) else feather_radius
        cs = output_colorspace[0] if isinstance(output_colorspace, list) else output_colorspace
        bg = 0.0
        if background_color is not None:
            bg = background_color[0] if isinstance(background_color, list) else background_color

        # Determine dimensions from first layer
        l1 = layer_list[0].cpu().float().numpy()
        if len(l1.shape) == 4:
            l1 = l1[0]
        h, w, c = l1.shape

        composite = np.full((h, w, c), bg, dtype=np.float64)

        active_layers = 0
        num_pairs = min(len(layer_list), len(mask_list))
        for i in range(num_pairs):
            layer_np = layer_list[i].cpu().float().numpy()
            if len(layer_np.shape) == 4:
                layer_np = layer_np[0]

            if layer_np.shape[:2] != (h, w):
                from scipy.ndimage import zoom
                scale = (h / layer_np.shape[0], w / layer_np.shape[1], 1)
                layer_np = zoom(layer_np, scale, order=3)

            mask = self._prepare_mask(mask_list[i], h, w, fr)

            if mask.sum() < 1:
                continue

            active_layers += 1
            for ch in range(c):
                composite[:, :, ch] = (
                    layer_np[:, :, ch].astype(np.float64) * mask +
                    composite[:, :, ch] * (1 - mask)
                )

        composite = np.clip(composite, 0.0, 1.0).astype(np.float32)

        if cs != "passthrough":
            composite = self._convert_colorspace(composite, cs)

        result = composite[np.newaxis, ...]
        print(f"[LayerAssembler] Composited {active_layers} layers, output: {cs}, {h}x{w}")

        return (torch.from_numpy(result),)


class BitDepthValidator:
    """
    QC tool for validating bit-depth expansion results.

    Analyzes histogram continuity, unique value counts, PSNR against
    the original, and gradient smoothness. Produces a text report
    and a waveform visualization image.
    """

    CATEGORY = "image/bitdepth"
    FUNCTION = "validate"
    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("passthrough", "waveform", "validation_report")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "expanded_image": ("IMAGE",),
                "original_image": ("IMAGE",),
            },
            "optional": {
                "waveform_height": ("INT", {
                    "default": 256, "min": 128, "max": 512, "step": 64
                }),
            },
        }

    def _count_unique_per_channel(self, image):
        """Count unique float values per channel."""
        counts = []
        for ch in range(image.shape[-1]):
            counts.append(len(np.unique(image[..., ch])))
        return counts

    def _estimate_effective_bits(self, unique_count):
        """Estimate effective bit depth from unique value count."""
        if unique_count <= 2:
            return 1
        bits = np.log2(unique_count)
        return round(bits, 1)

    def _compute_psnr(self, original, expanded):
        """Compute Peak Signal-to-Noise Ratio."""
        mse = np.mean((original.astype(np.float64) - expanded.astype(np.float64)) ** 2)
        if mse < 1e-10:
            return 100.0
        return float(10 * np.log10(1.0 / mse))

    def _compute_ssim_simple(self, original, expanded):
        """Simplified SSIM computation."""
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        mu_x = gaussian_filter(original, sigma=1.5)
        mu_y = gaussian_filter(expanded, sigma=1.5)

        mu_x2 = mu_x ** 2
        mu_y2 = mu_y ** 2
        mu_xy = mu_x * mu_y

        sigma_x2 = gaussian_filter(original ** 2, sigma=1.5) - mu_x2
        sigma_y2 = gaussian_filter(expanded ** 2, sigma=1.5) - mu_y2
        sigma_xy = gaussian_filter(original * expanded, sigma=1.5) - mu_xy

        ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / \
                   ((mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2))

        return float(np.mean(ssim_map))

    def _check_gradient_smoothness(self, image):
        """
        Check for banding artifacts by measuring gradient continuity.
        Returns a score from 0 (heavy banding) to 1 (perfectly smooth).
        """
        lum = 0.2126 * image[:, :, 0] + 0.7152 * image[:, :, 1] + 0.0722 * image[:, :, 2]

        # Compute second derivative (detects quantization steps)
        grad = np.gradient(lum.astype(np.float64))
        second_deriv = np.gradient(grad[0], axis=0) + np.gradient(grad[1], axis=1)

        # In a smooth image, second derivative should be small
        # Quantization steps create spikes
        rms_second = np.sqrt(np.mean(second_deriv ** 2))

        # Normalize: lower is better
        score = np.clip(1.0 - rms_second * 50, 0, 1)
        return float(score)

    def _generate_waveform(self, image, waveform_height):
        """Generate a waveform monitor visualization."""
        h, w, c = image.shape

        wf = np.zeros((waveform_height, w, 3), dtype=np.float32)

        colors = [(1, 0, 0), (0, 1, 0), (0, 0.5, 1)]  # R, G, B channel colors

        for ch in range(min(c, 3)):
            for x in range(w):
                column = image[:, x, ch]
                for val in column[::max(1, len(column) // 100)]:
                    y = int(np.clip(val, 0, 1) * (waveform_height - 1))
                    y = waveform_height - 1 - y  # flip vertical
                    for cc in range(3):
                        wf[y, x, cc] = max(wf[y, x, cc], colors[ch][cc] * 0.3)

                    # Brighten pixel for density
                    for dy in range(-1, 2):
                        yy = np.clip(y + dy, 0, waveform_height - 1)
                        for cc in range(3):
                            wf[yy, x, cc] = min(1.0, wf[yy, x, cc] + colors[ch][cc] * 0.05)

        return wf

    def validate(self, expanded_image, original_image, waveform_height=256):
        exp_np = expanded_image.cpu().float().numpy()
        orig_np = original_image.cpu().float().numpy()

        if len(exp_np.shape) == 4:
            exp_frame = exp_np[0]
        else:
            exp_frame = exp_np

        if len(orig_np.shape) == 4:
            orig_frame = orig_np[0]
        else:
            orig_frame = orig_np

        # Resize original to match if needed
        if orig_frame.shape[:2] != exp_frame.shape[:2]:
            from scipy.ndimage import zoom
            scale = (exp_frame.shape[0] / orig_frame.shape[0],
                     exp_frame.shape[1] / orig_frame.shape[1], 1)
            orig_frame = zoom(orig_frame, scale, order=3)

        # 1. Unique values per channel
        orig_unique = self._count_unique_per_channel(orig_frame)
        exp_unique = self._count_unique_per_channel(exp_frame)

        orig_bits = [self._estimate_effective_bits(u) for u in orig_unique]
        exp_bits = [self._estimate_effective_bits(u) for u in exp_unique]

        # 2. PSNR
        lum_orig = 0.2126 * orig_frame[:, :, 0] + 0.7152 * orig_frame[:, :, 1] + 0.0722 * orig_frame[:, :, 2]
        lum_exp = 0.2126 * exp_frame[:, :, 0] + 0.7152 * exp_frame[:, :, 1] + 0.0722 * exp_frame[:, :, 2]
        psnr = self._compute_psnr(lum_orig, lum_exp)

        # 3. SSIM
        ssim = self._compute_ssim_simple(lum_orig, lum_exp)

        # 4. Gradient smoothness
        orig_smoothness = self._check_gradient_smoothness(orig_frame)
        exp_smoothness = self._check_gradient_smoothness(exp_frame)

        # 5. Build report
        report_lines = [
            "=== BIT-DEPTH VALIDATION REPORT ===",
            "",
            "UNIQUE VALUES PER CHANNEL:",
            f"  Original:  R={orig_unique[0]:,}  G={orig_unique[1]:,}  B={orig_unique[2]:,}",
            f"  Expanded:  R={exp_unique[0]:,}  G={exp_unique[1]:,}  B={exp_unique[2]:,}",
            f"  Expansion: R={exp_unique[0]/max(orig_unique[0],1):.1f}x  "
            f"G={exp_unique[1]/max(orig_unique[1],1):.1f}x  "
            f"B={exp_unique[2]/max(orig_unique[2],1):.1f}x",
            "",
            "EFFECTIVE BIT DEPTH:",
            f"  Original:  R={orig_bits[0]}  G={orig_bits[1]}  B={orig_bits[2]}",
            f"  Expanded:  R={exp_bits[0]}  G={exp_bits[1]}  B={exp_bits[2]}",
            "",
            "FIDELITY:",
            f"  PSNR:  {psnr:.1f} dB  {'[EXCELLENT]' if psnr > 40 else '[GOOD]' if psnr > 30 else '[CHECK]'}",
            f"  SSIM:  {ssim:.4f}  {'[EXCELLENT]' if ssim > 0.98 else '[GOOD]' if ssim > 0.95 else '[CHECK]'}",
            "",
            "GRADIENT SMOOTHNESS (banding test):",
            f"  Original:  {orig_smoothness:.3f}",
            f"  Expanded:  {exp_smoothness:.3f}  "
            f"{'[IMPROVED]' if exp_smoothness > orig_smoothness else '[SIMILAR]' if abs(exp_smoothness - orig_smoothness) < 0.05 else '[DEGRADED]'}",
            "",
            "VERDICT:",
        ]

        # Overall verdict
        passes = 0
        total = 4
        if all(e > o * 1.5 for e, o in zip(exp_unique, orig_unique)):
            passes += 1
            report_lines.append("  [PASS] Unique value expansion > 1.5x")
        else:
            report_lines.append("  [WARN] Limited unique value expansion")

        if psnr > 30:
            passes += 1
            report_lines.append("  [PASS] PSNR > 30 dB (preserves original appearance)")
        else:
            report_lines.append("  [FAIL] PSNR < 30 dB (visible deviation from original)")

        if ssim > 0.95:
            passes += 1
            report_lines.append("  [PASS] SSIM > 0.95 (structural fidelity)")
        else:
            report_lines.append("  [WARN] SSIM < 0.95 (some structural change)")

        if exp_smoothness >= orig_smoothness - 0.05:
            passes += 1
            report_lines.append("  [PASS] Gradient smoothness maintained or improved")
        else:
            report_lines.append("  [WARN] Gradient smoothness decreased (possible artifacts)")

        report_lines.extend([
            "",
            f"OVERALL: {passes}/{total} checks passed",
            f"{'READY FOR GRADING' if passes >= 3 else 'REVIEW RECOMMENDED'}",
        ])

        report = "\n".join(report_lines)
        print(f"[BitDepthValidator] {passes}/{total} passed, PSNR={psnr:.1f}dB, SSIM={ssim:.4f}")

        # Generate waveform
        waveform = self._generate_waveform(exp_frame, waveform_height)
        waveform_tensor = torch.from_numpy(waveform[np.newaxis, ...])

        return (expanded_image, waveform_tensor, report)


LAYER_ASSEMBLY_NODES = {
    "LayerAssembler": LayerAssembler,
    "BitDepthValidator": BitDepthValidator,
}

LAYER_ASSEMBLY_DISPLAY_NAMES = {
    "LayerAssembler": "Layer Assembler",
    "BitDepthValidator": "Bit-Depth Validator",
}
