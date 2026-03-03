"""
High Bit-Depth Video Encoder for ComfyUI.

Encodes image batches (32-bit float tensors) to video files using ffmpeg,
preserving maximum bit depth throughout the pipeline.

Supported codecs:
  - FFV1 (16-bit lossless, MKV) — true high-bit-depth archival
  - ProRes 4444 (10-bit via ffmpeg, MOV) — NLE editing standard
  - ProRes 4444 XQ (10-bit via ffmpeg, MOV) — highest ProRes quality
  - H.265 HDR10 (10-bit, MP4) — broadcast/web delivery
  - AV1 (12-bit, MKV) — next-gen compression
"""

import os
import subprocess
import shutil
import struct
import numpy as np
import torch
import folder_paths


def _find_ffmpeg():
    return shutil.which("ffmpeg")


class HighBitDepthVideoEncoder:
    """Encode image batch to high-bit-depth video via raw float pipe to ffmpeg."""

    CATEGORY = "image/video"
    FUNCTION = "encode"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    OUTPUT_NODE = True

    CODECS = [
        "FFV1 16-bit Lossless (MKV)",
        "ProRes 4444 (MOV)",
        "ProRes 4444 XQ (MOV)",
        "H.265 HDR10 (MP4)",
        "AV1 12-bit (MKV)",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "video_output"}),
                "codec": (cls.CODECS, {"default": "FFV1 16-bit Lossless (MKV)"}),
                "frame_rate": ("FLOAT", {
                    "default": 24.0, "min": 1.0, "max": 120.0, "step": 0.001,
                }),
            },
            "optional": {
                "quality": ("INT", {
                    "default": 5, "min": 0, "max": 31, "step": 1,
                    "tooltip": "Quality (lower=better). Used by ProRes (0-13) and H.265 CRF (0-51).",
                }),
            },
        }

    def _build_ffmpeg_args(self, codec, w, h, fps, output_path, quality):
        ffmpeg = _find_ffmpeg()
        if not ffmpeg:
            raise RuntimeError("ffmpeg not found. Install ffmpeg to encode video.")

        base = [
            ffmpeg, "-y",
            "-f", "rawvideo",
            "-pix_fmt", "gbrpf32le",
            "-s", f"{w}x{h}",
            "-r", str(fps),
            "-i", "pipe:0",
        ]

        if "FFV1" in codec:
            return base + [
                "-pix_fmt", "gbrp16le",
                "-c:v", "ffv1",
                "-level", "3",
                "-coder", "1",
                "-context", "1",
                "-g", "1",
                "-slices", "4",
                output_path,
            ]
        elif "4444 XQ" in codec:
            return base + [
                "-pix_fmt", "yuv444p10le",
                "-c:v", "prores_ks",
                "-profile:v", "5",
                "-vendor", "apl0",
                "-qscale:v", str(min(quality, 13)),
                output_path,
            ]
        elif "4444" in codec:
            return base + [
                "-pix_fmt", "yuv444p10le",
                "-c:v", "prores_ks",
                "-profile:v", "4",
                "-vendor", "apl0",
                "-qscale:v", str(min(quality, 13)),
                output_path,
            ]
        elif "H.265" in codec:
            return base + [
                "-pix_fmt", "yuv444p10le",
                "-c:v", "libx265",
                "-preset", "slow",
                "-crf", str(quality),
                "-x265-params", "colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc",
                "-tag:v", "hvc1",
                output_path,
            ]
        elif "AV1" in codec:
            return base + [
                "-pix_fmt", "yuv444p12le",
                "-c:v", "libaom-av1",
                "-crf", str(quality),
                "-cpu-used", "4",
                "-row-mt", "1",
                output_path,
            ]

        raise ValueError(f"Unknown codec: {codec}")

    def _get_extension(self, codec):
        if "MKV" in codec:
            return ".mkv"
        elif "MOV" in codec:
            return ".mov"
        elif "MP4" in codec:
            return ".mp4"
        return ".mkv"

    def _tensor_to_gbrp_float32(self, frame_tensor):
        """Convert HWC RGB float32 tensor to planar GBR float32 bytes for ffmpeg."""
        frame = frame_tensor.cpu().numpy().astype(np.float32)
        g = np.ascontiguousarray(frame[:, :, 1])
        b = np.ascontiguousarray(frame[:, :, 2])
        r = np.ascontiguousarray(frame[:, :, 0])
        return g.tobytes() + b.tobytes() + r.tobytes()

    def encode(self, images, filename_prefix, codec, frame_rate, quality=5):
        n, h, w, c = images.shape
        ext = self._get_extension(codec)

        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), w, h
        )
        os.makedirs(full_output_folder, exist_ok=True)
        output_path = os.path.join(full_output_folder, f"{filename}_{counter:05d}{ext}")

        args = self._build_ffmpeg_args(codec, w, h, frame_rate, output_path, quality)

        print(f"[HighBitDepthVideoEncoder] Encoding {n} frames @ {w}x{h} → {codec}")
        print(f"  Output: {output_path}")

        proc = subprocess.Popen(
            args, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )

        for i in range(n):
            raw = self._tensor_to_gbrp_float32(images[i])
            proc.stdin.write(raw)
            if (i + 1) % 10 == 0 or i == n - 1:
                print(f"  Frame {i + 1}/{n}")

        proc.stdin.close()
        stdout, stderr = proc.communicate()

        if proc.returncode != 0:
            error_msg = stderr.decode("utf-8", errors="replace")[-2000:]
            print(f"[HighBitDepthVideoEncoder] ffmpeg error:\n{error_msg}")
            raise RuntimeError(f"ffmpeg failed with code {proc.returncode}")

        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"[HighBitDepthVideoEncoder] Done: {output_path} ({file_size:.1f} MB)")

        return {"ui": {"text": [f"{codec}: {file_size:.1f} MB"]},
                "result": (output_path,)}


VIDEO_ENCODER_NODES = {
    "HighBitDepthVideoEncoder": HighBitDepthVideoEncoder,
}

VIDEO_ENCODER_DISPLAY_NAMES = {
    "HighBitDepthVideoEncoder": "High Bit-Depth Video Encoder",
}
