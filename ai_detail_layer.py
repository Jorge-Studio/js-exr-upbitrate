"""
AI Detail Enhancement nodes for ComfyUI.

LayerDetailEnhancer: Uses Laplacian pyramid frequency-domain blending to
inject AI-generated micro-detail into a layer while preserving the original
color grading. The low frequencies (tone, color, exposure) stay from the
original; only high-frequency texture detail comes from the AI pass.

This node works standalone without a diffusion model -- it can accept
any pre-processed "detail image" from ControlNet Tile, upscaler, or
manual texture overlay. It also includes a built-in sharpening mode
that doesn't require any AI model.
"""

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from .fractal_utils import laplacian_pyramid, reconstruct_from_pyramid


class LayerDetailEnhancer:
    """
    Enhance micro-detail in a layer using frequency-domain blending.

    Accepts an optional AI-generated detail image. If not provided,
    uses built-in unsharp-mask + fractal detail synthesis.

    Key innovation: low frequencies from original (preserves grading),
    high frequencies from detail source (adds texture).
    """

    CATEGORY = "image/bitdepth"
    FUNCTION = "enhance"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("enhanced",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "layer_image": ("IMAGE",),
                "detail_strength": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "How much AI/synthetic detail to blend in. "
                               "0.0 = original only, 1.0 = full detail replacement."
                }),
                "frequency_cutoff": ("FLOAT", {
                    "default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Which frequency bands to blend. "
                               "0.0 = blend all bands, 1.0 = only highest frequencies."
                }),
                "pyramid_levels": ("INT", {
                    "default": 5, "min": 2, "max": 8, "step": 1,
                    "tooltip": "Number of frequency decomposition levels."
                }),
                "mode": (["auto_enhance", "detail_image", "sharpen_only"], {
                    "default": "auto_enhance",
                    "tooltip": "auto_enhance: fractal detail synthesis. "
                               "detail_image: blend from provided image. "
                               "sharpen_only: unsharp mask, no synthesis."
                }),
            },
            "optional": {
                "layer_mask": ("MASK",),
                "detail_image": ("IMAGE",),
                "text_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Description of expected detail (for future AI integration). "
                               "Currently used as metadata tag."
                }),
            },
        }

    def _auto_enhance_detail(self, frame, strength):
        """
        Generate synthetic micro-detail using multi-scale edge enhancement
        and fractal noise injection. No AI model needed.
        """
        h, w, c = frame.shape
        enhanced = frame.copy().astype(np.float64)

        for ch in range(c):
            channel = frame[:, :, ch].astype(np.float64)

            # Multi-scale unsharp mask
            blur_fine = gaussian_filter(channel, sigma=1.0)
            blur_mid = gaussian_filter(channel, sigma=3.0)
            blur_coarse = gaussian_filter(channel, sigma=7.0)

            detail_fine = channel - blur_fine
            detail_mid = blur_fine - blur_mid
            detail_coarse = blur_mid - blur_coarse

            # Enhance fine detail, preserve mid, slightly enhance coarse
            enhanced[:, :, ch] = (
                blur_coarse +
                detail_coarse * (1.0 + strength * 0.1) +
                detail_mid * (1.0 + strength * 0.3) +
                detail_fine * (1.0 + strength * 0.8)
            )

        return np.clip(enhanced, 0.0, 1.0).astype(np.float32)

    def _sharpen(self, frame, strength):
        """Simple unsharp mask sharpening."""
        result = frame.copy().astype(np.float64)

        for ch in range(frame.shape[2]):
            blurred = gaussian_filter(frame[:, :, ch].astype(np.float64), sigma=1.5)
            detail = frame[:, :, ch].astype(np.float64) - blurred
            result[:, :, ch] = frame[:, :, ch].astype(np.float64) + detail * strength * 2.0

        return np.clip(result, 0.0, 1.0).astype(np.float32)

    def _frequency_blend(self, original, detail_source, frequency_cutoff,
                         detail_strength, pyramid_levels):
        """
        Blend using Laplacian pyramid.

        Original provides low frequencies (tone, color, exposure).
        Detail source provides high frequencies (texture, micro-detail).
        """
        h, w, c = original.shape
        result = np.zeros_like(original, dtype=np.float32)

        for ch in range(c):
            orig_pyr = laplacian_pyramid(original[:, :, ch], levels=pyramid_levels)
            detail_pyr = laplacian_pyramid(detail_source[:, :, ch], levels=pyramid_levels)

            blended_pyr = []
            for level_idx in range(len(orig_pyr)):
                # Level 0 = highest frequency, last = lowest (residual)
                # frequency_cutoff determines which levels get detail blending
                level_position = level_idx / max(len(orig_pyr) - 1, 1)

                if level_idx == len(orig_pyr) - 1:
                    # Residual (DC component): always from original
                    blended_pyr.append(orig_pyr[level_idx])
                elif level_position <= (1 - frequency_cutoff):
                    # High frequency levels: blend in detail
                    blend_amount = detail_strength * (1.0 - level_position)
                    blended = (
                        orig_pyr[level_idx] * (1 - blend_amount) +
                        detail_pyr[level_idx] * blend_amount
                    )
                    blended_pyr.append(blended)
                else:
                    # Low frequency levels: keep original
                    blended_pyr.append(orig_pyr[level_idx])

            result[:, :, ch] = reconstruct_from_pyramid(blended_pyr)

        return np.clip(result, 0.0, 1.0).astype(np.float32)

    def enhance(self, layer_image, detail_strength, frequency_cutoff,
                pyramid_levels, mode, layer_mask=None, detail_image=None,
                text_prompt=""):

        img_np = layer_image.cpu().float().numpy()
        if len(img_np.shape) == 3:
            img_np = img_np[np.newaxis, ...]

        # Prepare mask
        mask = None
        if layer_mask is not None:
            mask = layer_mask.cpu().float().numpy()
            if len(mask.shape) == 3:
                mask = mask[0] if mask.shape[0] == 1 else mask.mean(axis=-1)

        # Prepare detail source if provided
        detail_np = None
        if detail_image is not None and mode == "detail_image":
            detail_np = detail_image.cpu().float().numpy()
            if len(detail_np.shape) == 3:
                detail_np = detail_np[np.newaxis, ...]

        results = []
        for frame_idx in range(img_np.shape[0]):
            frame = img_np[frame_idx]
            h, w, c = frame.shape

            if mode == "sharpen_only":
                enhanced = self._sharpen(frame, detail_strength)
            elif mode == "detail_image" and detail_np is not None:
                det_frame = detail_np[min(frame_idx, detail_np.shape[0] - 1)]
                # Resize detail to match if needed
                if det_frame.shape[:2] != (h, w):
                    from scipy.ndimage import zoom
                    scale_h = h / det_frame.shape[0]
                    scale_w = w / det_frame.shape[1]
                    det_frame = zoom(det_frame, (scale_h, scale_w, 1), order=3)

                enhanced = self._frequency_blend(
                    frame, det_frame, frequency_cutoff,
                    detail_strength, pyramid_levels
                )
            else:
                # Auto-enhance: generate synthetic detail then frequency blend
                detail_synth = self._auto_enhance_detail(frame, detail_strength)
                enhanced = self._frequency_blend(
                    frame, detail_synth, frequency_cutoff,
                    detail_strength, pyramid_levels
                )

            # Apply mask if provided
            if mask is not None:
                m = mask
                if m.shape != (h, w):
                    from scipy.ndimage import zoom
                    m = zoom(m, (h / m.shape[0], w / m.shape[1]), order=1)
                for ch in range(c):
                    enhanced[:, :, ch] = frame[:, :, ch] * (1 - m) + enhanced[:, :, ch] * m

            results.append(enhanced)

        output = np.stack(results, axis=0)

        prompt_tag = f" prompt='{text_prompt[:40]}'" if text_prompt else ""
        print(f"[LayerDetailEnhancer] mode={mode}, strength={detail_strength}, "
              f"cutoff={frequency_cutoff}, levels={pyramid_levels}{prompt_tag}")

        return (torch.from_numpy(output),)


AI_DETAIL_NODES = {
    "LayerDetailEnhancer": LayerDetailEnhancer,
}

AI_DETAIL_DISPLAY_NAMES = {
    "LayerDetailEnhancer": "Layer Detail Enhancer",
}
