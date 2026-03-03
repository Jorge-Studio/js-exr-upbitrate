"""
Temporal Fractal Bit-Depth Expander for ComfyUI.

Applies fractal bit-depth expansion across video frames with
motion-compensated noise seeding to prevent temporal "swimming"
of added detail.

Unlike the single-frame FractalBitDepthExpander, this node:
  - Analyzes optical flow between adjacent frames
  - Warps the fractal noise field to follow motion
  - Ensures added detail stays locked to scene content
"""

import numpy as np
import torch

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

from .fractal_utils import (
    compute_local_fractal_dimension,
    fractal_brownian_motion,
    hermite_interpolate_neighbors,
    sobel_gradient,
)


def _compute_optical_flow(prev_gray, curr_gray):
    """Compute dense optical flow between two grayscale frames."""
    if not _HAS_CV2:
        return np.zeros((*prev_gray.shape, 2), dtype=np.float32)

    prev_u8 = (np.clip(prev_gray, 0, 1) * 255).astype(np.uint8)
    curr_u8 = (np.clip(curr_gray, 0, 1) * 255).astype(np.uint8)

    flow = cv2.calcOpticalFlowFarneback(
        prev_u8, curr_u8,
        None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )
    return flow


def _warp_image(img, flow):
    """Warp an image using optical flow (backward warp)."""
    if not _HAS_CV2:
        return img

    h, w = img.shape[:2]
    map_x, map_y = np.meshgrid(np.arange(w, dtype=np.float32),
                                np.arange(h, dtype=np.float32))
    map_x = map_x + flow[:, :, 0]
    map_y = map_y + flow[:, :, 1]

    if len(img.shape) == 2:
        return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT)
    else:
        return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT)


class TemporalFractalExpander:
    """
    Motion-compensated fractal bit-depth expansion for video.

    Generates a fractal noise field on the first frame, then warps it
    to follow motion in subsequent frames, preventing temporal artifacts.
    """

    CATEGORY = "image/bitdepth"
    FUNCTION = "expand"
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("expanded", "fractal_map")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "target_bit_depth": (["16", "24", "32"], {"default": "32"}),
                "fractal_octaves": ("INT", {
                    "default": 4, "min": 1, "max": 8, "step": 1,
                }),
                "fractal_persistence": ("FLOAT", {
                    "default": 0.5, "min": 0.1, "max": 0.9, "step": 0.05,
                }),
                "gradient_smoothness": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05,
                }),
                "temporal_coherence": ("FLOAT", {
                    "default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "How strongly to lock fractal detail to motion. 1.0 = fully locked.",
                }),
                "motion_compensation": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use optical flow to track motion. Disable for static scenes.",
                }),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**31 - 1}),
            },
        }

    def _frame_to_gray(self, frame):
        return 0.2126 * frame[:, :, 0] + 0.7152 * frame[:, :, 1] + 0.0722 * frame[:, :, 2]

    def _expand_frame(self, frame, fbm_noise, lfd_map, grad_direction,
                      gradient_smoothness):
        """Expand one frame using pre-computed (motion-warped) noise."""
        h, w, c = frame.shape
        step = 1.0 / 255.0
        expanded = np.zeros_like(frame, dtype=np.float64)
        fractal_scale = np.clip(lfd_map / 2.0, 0, 1)

        for ch in range(c):
            channel = frame[:, :, ch].astype(np.float64)
            gradient_offset = grad_direction * step * 0.4
            fractal_offset = fbm_noise * step * 0.3 * fractal_scale
            ch_smooth = hermite_interpolate_neighbors(frame, channel=ch)
            smooth_off = ch_smooth * step * 0.3

            blend = fractal_scale * (1 - gradient_smoothness) + \
                    (1 - fractal_scale) * gradient_smoothness
            fractal_weight = 1 - blend
            smooth_weight = blend
            offset = smooth_off * smooth_weight + fractal_offset * fractal_weight + gradient_offset
            expanded[:, :, ch] = channel + offset

        return np.clip(expanded, 0.0, 1.0).astype(np.float32)

    def expand(self, images, target_bit_depth, fractal_octaves, fractal_persistence,
               gradient_smoothness, temporal_coherence, motion_compensation, seed):

        img_np = images.cpu().float().numpy()
        if len(img_np.shape) == 3:
            img_np = img_np[np.newaxis, ...]

        n, h, w, c = img_np.shape
        print(f"[TemporalFractalExpander] Processing {n} frames @ {w}x{h}")

        base_fbm = fractal_brownian_motion(
            (h, w), octaves=fractal_octaves,
            persistence=fractal_persistence, seed=seed,
        )

        expanded_frames = []
        lfd_frames = []
        prev_gray = None
        current_fbm = base_fbm.copy()

        for i in range(n):
            frame = img_np[i]
            gray = self._frame_to_gray(frame)
            lfd_map = compute_local_fractal_dimension(gray, patch_size=7)
            grad_x, grad_y = sobel_gradient(gray)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            grad_norm = grad_mag.max()
            grad_direction = (grad_x + grad_y) / (grad_norm * 2) if grad_norm > 0 else np.zeros_like(grad_x)

            if motion_compensation and prev_gray is not None and _HAS_CV2:
                flow = _compute_optical_flow(prev_gray, gray)
                warped_fbm = _warp_image(current_fbm, flow)
                fresh_fbm = fractal_brownian_motion(
                    (h, w), octaves=fractal_octaves,
                    persistence=fractal_persistence, seed=seed + i * 997,
                )
                current_fbm = warped_fbm * temporal_coherence + \
                              fresh_fbm * (1 - temporal_coherence)
            elif i > 0 and not motion_compensation:
                blend = temporal_coherence
                fresh_fbm = fractal_brownian_motion(
                    (h, w), octaves=fractal_octaves,
                    persistence=fractal_persistence, seed=seed + i * 997,
                )
                current_fbm = base_fbm * blend + fresh_fbm * (1 - blend)

            exp = self._expand_frame(frame, current_fbm, lfd_map, grad_direction,
                                     gradient_smoothness)
            expanded_frames.append(exp)

            lfd_vis = (lfd_map - lfd_map.min()) / (lfd_map.max() - lfd_map.min() + 1e-10)
            lfd_frames.append(np.stack([lfd_vis, lfd_vis, lfd_vis], axis=-1).astype(np.float32))

            prev_gray = gray

            if (i + 1) % 10 == 0 or i == n - 1:
                print(f"  Frame {i + 1}/{n}")

        expanded = np.stack(expanded_frames, axis=0)
        lfd_stack = np.stack(lfd_frames, axis=0)

        u_before = len(np.unique((img_np[0, :, :, 0] * 255).astype(np.uint8)))
        u_after = len(np.unique(expanded[0, :, :, 0]))
        print(f"[TemporalFractalExpander] {u_before} → {u_after} unique values/channel "
              f"(target: {target_bit_depth}-bit)")

        return (torch.from_numpy(expanded), torch.from_numpy(lfd_stack))


TEMPORAL_FRACTAL_NODES = {
    "TemporalFractalExpander": TemporalFractalExpander,
}

TEMPORAL_FRACTAL_DISPLAY_NAMES = {
    "TemporalFractalExpander": "Temporal Fractal Bit-Depth Expander",
}
