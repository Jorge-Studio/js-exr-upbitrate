"""
Frame Interpolator for ComfyUI.

Converts video frame rates (e.g., 16fps model output → 24fps cinema)
by generating intermediate frames using optical flow warping or
simple blending.
"""

import numpy as np
import torch

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


def _flow_interpolate(frame_a, frame_b, t):
    """Generate an intermediate frame at position t (0-1) between a and b using optical flow."""
    if not _HAS_CV2:
        return _blend_interpolate(frame_a, frame_b, t)

    gray_a = (np.clip(frame_a.mean(axis=-1), 0, 1) * 255).astype(np.uint8)
    gray_b = (np.clip(frame_b.mean(axis=-1), 0, 1) * 255).astype(np.uint8)

    flow_ab = cv2.calcOpticalFlowFarneback(
        gray_a, gray_b, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )

    h, w = frame_a.shape[:2]
    map_x, map_y = np.meshgrid(np.arange(w, dtype=np.float32),
                                np.arange(h, dtype=np.float32))

    map_x_a = map_x + flow_ab[:, :, 0] * t
    map_y_a = map_y + flow_ab[:, :, 1] * t
    warped_a = cv2.remap(frame_a, map_x_a, map_y_a, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT)

    map_x_b = map_x - flow_ab[:, :, 0] * (1 - t)
    map_y_b = map_y - flow_ab[:, :, 1] * (1 - t)
    warped_b = cv2.remap(frame_b, map_x_b, map_y_b, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT)

    return (warped_a * (1 - t) + warped_b * t).astype(np.float32)


def _blend_interpolate(frame_a, frame_b, t):
    """Simple linear blend between two frames."""
    return (frame_a * (1 - t) + frame_b * t).astype(np.float32)


class FrameInterpolator:
    """Interpolate video frames to change frame rate (e.g., 16fps → 24fps)."""

    CATEGORY = "image/video"
    FUNCTION = "interpolate"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "source_fps": ("FLOAT", {
                    "default": 16.0, "min": 1.0, "max": 120.0, "step": 0.001,
                    "tooltip": "Original frame rate of the input video.",
                }),
                "target_fps": ("FLOAT", {
                    "default": 24.0, "min": 1.0, "max": 120.0, "step": 0.001,
                    "tooltip": "Desired output frame rate.",
                }),
                "method": (["optical_flow", "blend"], {
                    "default": "optical_flow",
                    "tooltip": "optical_flow = motion-aware (requires cv2). blend = simple crossfade.",
                }),
            },
        }

    def interpolate(self, images, source_fps, target_fps, method):
        img_np = images.cpu().float().numpy()
        if len(img_np.shape) == 3:
            img_np = img_np[np.newaxis, ...]

        n_in = img_np.shape[0]

        if n_in < 2:
            return (images,)

        duration = (n_in - 1) / source_fps
        n_out = max(2, int(round(duration * target_fps)) + 1)
        ratio = source_fps / target_fps

        print(f"[FrameInterpolator] {n_in} frames @ {source_fps}fps → {n_out} frames @ {target_fps}fps "
              f"(method: {method})")

        interp_fn = _flow_interpolate if method == "optical_flow" else _blend_interpolate
        output_frames = []

        for out_i in range(n_out):
            src_pos = out_i * ratio
            idx_a = int(np.floor(src_pos))
            idx_b = min(idx_a + 1, n_in - 1)
            t = src_pos - idx_a

            idx_a = min(idx_a, n_in - 1)

            if idx_a == idx_b or t < 1e-6:
                output_frames.append(img_np[idx_a])
            elif t > 1 - 1e-6:
                output_frames.append(img_np[idx_b])
            else:
                interped = interp_fn(img_np[idx_a], img_np[idx_b], t)
                output_frames.append(interped)

            if (out_i + 1) % 20 == 0 or out_i == n_out - 1:
                print(f"  Frame {out_i + 1}/{n_out}")

        result = np.stack(output_frames, axis=0)
        return (torch.from_numpy(result),)


FRAME_INTERP_NODES = {
    "FrameInterpolator": FrameInterpolator,
}

FRAME_INTERP_DISPLAY_NAMES = {
    "FrameInterpolator": "Frame Interpolator",
}
