"""
Fractal Bit-Depth Expansion nodes for ComfyUI.

FractalBitDepthExpander: Expands 8-bit quantized images to 32-bit float
using Local Fractal Dimension analysis, gradient-aware sub-pixel placement,
and fractal Brownian motion micro-texture.

PerceptualDither: Applies perceptual dithering (blue noise / TPDF / fractal)
as a final pass at the target bit-depth boundary.
"""

import numpy as np
import torch

from .fractal_utils import (
    compute_local_fractal_dimension,
    fractal_brownian_motion,
    hermite_interpolate_neighbors,
    rational_fractal_cubic_spline,
    generate_blue_noise,
    generate_tpdf_dither,
    sobel_gradient,
    temporal_coherent_seed,
)


class FractalBitDepthExpander:
    """
    Expand 8-bit images to 32-bit float with fractal-structured tonal fill.

    Uses Local Fractal Dimension to adapt the fill strategy per-pixel:
    smooth Hermite splines for gradients, fBm micro-texture for detail areas.
    """

    CATEGORY = "image/bitdepth"
    FUNCTION = "expand"
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("expanded", "fractal_map")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_bit_depth": (["16", "24", "32"], {"default": "32"}),
                "fractal_octaves": ("INT", {
                    "default": 4, "min": 1, "max": 8, "step": 1,
                    "tooltip": "Detail levels in fractal noise. More = finer micro-texture."
                }),
                "fractal_persistence": ("FLOAT", {
                    "default": 0.5, "min": 0.1, "max": 0.9, "step": 0.05,
                    "tooltip": "Amplitude decay per octave. Lower = smoother fill."
                }),
                "gradient_smoothness": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Bias toward smooth (1.0) vs textured (0.0) fill."
                }),
                "temporal_coherence": ("FLOAT", {
                    "default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Frame-to-frame stability for video. 1.0 = static noise."
                }),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**31 - 1}),
            },
        }

    def _expand_frame(self, frame_np, fractal_octaves, fractal_persistence,
                      gradient_smoothness, temporal_coherence, seed,
                      frame_idx=0):
        """Expand a single frame from 8-bit quantized to 32-bit float."""
        h, w, c = frame_np.shape

        # 1. Convert to luminance for analysis
        lum = 0.2126 * frame_np[:, :, 0] + 0.7152 * frame_np[:, :, 1] + 0.0722 * frame_np[:, :, 2]

        # 2. Compute Local Fractal Dimension
        lfd_map = compute_local_fractal_dimension(lum, patch_size=7)

        # 3. Compute gradient field
        grad_x, grad_y = sobel_gradient(lum)
        grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

        # Normalize gradient direction for offset computation
        grad_norm = grad_magnitude.max()
        if grad_norm > 0:
            grad_direction = (grad_x + grad_y) / (grad_norm * 2)
        else:
            grad_direction = np.zeros_like(grad_x)

        # 4. Generate temporally coherent fractal noise
        frame_seed = temporal_coherent_seed(seed, frame_idx, temporal_coherence)
        fbm = fractal_brownian_motion(
            (h, w), octaves=fractal_octaves,
            persistence=fractal_persistence, seed=frame_seed
        )

        # 5. Compute Hermite smooth offset
        smooth_offset = hermite_interpolate_neighbors(frame_np)

        # 6. Size of one 8-bit quantization step
        step = 1.0 / 255.0

        # 7. Build per-channel expanded image
        expanded = np.zeros_like(frame_np, dtype=np.float64)

        for ch in range(c):
            channel = frame_np[:, :, ch].astype(np.float64)

            # Gradient-based sub-pixel offset: push toward brighter neighbor
            gradient_offset = grad_direction * step * 0.4

            # Fractal micro-texture scaled by LFD
            # High LFD regions get more fractal detail
            fractal_scale = np.clip(lfd_map / 2.0, 0, 1)
            fractal_offset = fbm * step * 0.3 * fractal_scale

            # Smooth Hermite offset for low-LFD regions
            ch_smooth = hermite_interpolate_neighbors(frame_np, channel=ch)
            smooth_off = ch_smooth * step * 0.3

            # Blend: low LFD = smooth, high LFD = fractal
            blend = fractal_scale * (1 - gradient_smoothness) + \
                    (1 - fractal_scale) * gradient_smoothness
            # blend near 1 = use smooth, blend near 0 = use fractal
            # Invert so smoothness parameter works intuitively
            fractal_weight = 1 - blend
            smooth_weight = blend

            offset = smooth_off * smooth_weight + fractal_offset * fractal_weight + gradient_offset

            expanded[:, :, ch] = channel + offset

        expanded = np.clip(expanded, 0.0, 1.0).astype(np.float32)

        # Normalize LFD map to [0,1] for visualization
        lfd_vis = (lfd_map - lfd_map.min()) / (lfd_map.max() - lfd_map.min() + 1e-10)
        lfd_vis_rgb = np.stack([lfd_vis, lfd_vis, lfd_vis], axis=-1).astype(np.float32)

        return expanded, lfd_vis_rgb

    def expand(self, image, target_bit_depth, fractal_octaves, fractal_persistence,
               gradient_smoothness, temporal_coherence, seed):

        img_np = image.cpu().float().numpy()
        if len(img_np.shape) == 3:
            img_np = img_np[np.newaxis, ...]

        expanded_frames = []
        lfd_frames = []

        for i in range(img_np.shape[0]):
            frame = img_np[i]
            exp, lfd_vis = self._expand_frame(
                frame, fractal_octaves, fractal_persistence,
                gradient_smoothness, temporal_coherence, seed,
                frame_idx=i
            )
            expanded_frames.append(exp)
            lfd_frames.append(lfd_vis)

        expanded = np.stack(expanded_frames, axis=0)
        lfd_stack = np.stack(lfd_frames, axis=0)

        bit_depth = int(target_bit_depth)
        unique_before = len(np.unique((img_np[0, :, :, 0] * 255).astype(np.uint8)))
        unique_after = len(np.unique(expanded[0, :, :, 0]))
        print(f"[FractalBitDepthExpander] {unique_before} -> {unique_after} unique values/channel "
              f"(target: {bit_depth}-bit, {img_np.shape[0]} frames)")

        return (torch.from_numpy(expanded), torch.from_numpy(lfd_stack))


class PerceptualDither:
    """
    Apply perceptual dithering as a final pass. Breaks any remaining
    quantization artifacts with perceptually invisible noise patterns.
    """

    CATEGORY = "image/bitdepth"
    FUNCTION = "dither"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("dithered",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "method": (["blue_noise", "TPDF", "fractal"],),
                "strength": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Dither intensity. 0.5 = standard, 1.0 = aggressive."
                }),
                "target_bits": (["10", "12", "14", "16", "32"], {
                    "default": "16",
                    "tooltip": "Target bit depth - dither is scaled to this precision."
                }),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**31 - 1}),
            },
        }

    def dither(self, image, method, strength, target_bits, seed):
        img_np = image.cpu().float().numpy()
        if len(img_np.shape) == 3:
            img_np = img_np[np.newaxis, ...]

        bits = int(target_bits)
        # Dither amplitude = one step at target bit depth
        step = 1.0 / (2 ** bits - 1)

        results = []
        for i in range(img_np.shape[0]):
            frame = img_np[i]
            h, w, c = frame.shape
            frame_seed = seed + i * 997

            if method == "blue_noise":
                noise = generate_blue_noise((h, w), seed=frame_seed)
                noise = (noise - 0.5) * 2  # center at 0
            elif method == "TPDF":
                noise = generate_tpdf_dither((h, w), seed=frame_seed)
            else:  # fractal
                noise = fractal_brownian_motion((h, w), octaves=3,
                                                persistence=0.5,
                                                seed=frame_seed)

            # Scale to target step size
            noise_scaled = noise * step * strength

            # Apply per-channel with slight variation
            dithered = frame.copy()
            for ch in range(c):
                ch_noise = noise_scaled * (1.0 + 0.05 * (ch - 1))
                dithered[:, :, ch] = frame[:, :, ch] + ch_noise

            results.append(np.clip(dithered, 0.0, 1.0).astype(np.float32))

        output = np.stack(results, axis=0)
        print(f"[PerceptualDither] Applied {method} dither at {bits}-bit precision, "
              f"strength={strength}")

        return (torch.from_numpy(output),)


FRACTAL_BITDEPTH_NODES = {
    "FractalBitDepthExpander": FractalBitDepthExpander,
    "PerceptualDither": PerceptualDither,
}

FRACTAL_BITDEPTH_DISPLAY_NAMES = {
    "FractalBitDepthExpander": "Fractal Bit-Depth Expander",
    "PerceptualDither": "Perceptual Dither",
}
