"""
High bit-depth export nodes for ComfyUI: 16/32-bit EXR and ProRes 4444 XQ.
Similar in spirit to Luminance-Stack-Processor — preserves dynamic range,
optional linear color and debanding for Image and Video.
"""
from __future__ import annotations

import os
import sys
import math
import tempfile
import subprocess
import shutil
from typing import Optional

import numpy as np
import torch
import folder_paths
from comfy_api.latest import ComfyExtension, IO, UI
from comfy.cli_args import args

# Optional: imageio for EXR, cv2 for debanding
try:
    import imageio
    _HAS_IMAGEIO = True
except ImportError:
    _HAS_IMAGEIO = False

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


# --- sRGB <-> linear (for HDR / correct EXR) ---
def _srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    out = torch.where(x <= 0.04045, x / 12.92, torch.pow((x + 0.055) / 1.055, 2.4))
    return out.clamp(0.0, None)


def _linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    out = torch.where(x <= 0.0031308, x * 12.92, 1.055 * torch.pow(x, 1.0 / 2.4) - 0.055)
    return out.clamp(0.0, 1.0)


def _deband_image_numpy(img_t: torch.Tensor, strength: float) -> torch.Tensor:
    """Simple debanding: bilateral filter (OpenCV supports 8u and 32f only). Returns tensor on same device as input."""
    if not _HAS_CV2 or strength <= 0:
        return img_t
    # (H,W,C) float 0-1, RGB; use float32 for cv2.bilateralFilter
    img = np.ascontiguousarray(img_t.clamp(0.0, 1.0).cpu().numpy().astype("float32"))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    sigma = max(1.0, 20.0 * strength)
    filtered = cv2.bilateralFilter(img_bgr, 5, sigma, sigma)
    out_rgb = cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)
    out = torch.from_numpy(out_rgb)
    return out.to(device=img_t.device, dtype=img_t.dtype)


def _tensor_to_numpy_rgb(images: torch.Tensor, index: int) -> "torch.Tensor":
    """Single frame (H,W,C) float 0-1, same device as images."""
    return images[index].clamp(0.0, 1.0)


def _find_ffmpeg() -> Optional[str]:
    return shutil.which("ffmpeg")


def _write_exr(filepath: str, data: "np.ndarray", compression: str = "zip") -> None:
    """Write float32/float16 array as EXR. Prefer OpenCV (reliable on Windows), then imageio.
    When data is float16, writes 16-bit EXR; when float32, writes 32-bit EXR."""
    # OpenCV expects float32 and uses IMWRITE_EXR_TYPE to choose 16- or 32-bit on disk
    if data.dtype == "float16":
        data_f32 = data.astype("float32")
        request_16bit = True
    else:
        data_f32 = np.ascontiguousarray(data.astype("float32"))
        request_16bit = False

    # 1) OpenCV: most reliable for EXR on Windows when opencv-python is installed
    if _HAS_CV2:
        try:
            # OpenCV expects BGR, (H,W,C) float32
            bgr = cv2.cvtColor(data_f32, cv2.COLOR_RGB2BGR)
            params = []
            exr_type_key = getattr(cv2, "IMWRITE_EXR_TYPE", None)
            if exr_type_key is not None:
                # IMWRITE_EXR_TYPE_HALF = 1 (16-bit), IMWRITE_EXR_TYPE_FLOAT = 2 (32-bit)
                exr_half = getattr(cv2, "IMWRITE_EXR_TYPE_HALF", 1)
                exr_float = getattr(cv2, "IMWRITE_EXR_TYPE_FLOAT", 2)
                params.append(exr_type_key)
                params.append(exr_half if request_16bit else exr_float)
            comp_key = getattr(cv2, "IMWRITE_EXR_COMPRESSION", None)
            if comp_key is not None:
                comp_map = {
                    "none": getattr(cv2, "IMWRITE_EXR_COMPRESSION_NO", 0),
                    "rle": getattr(cv2, "IMWRITE_EXR_COMPRESSION_RLE", 1),
                    "zips": getattr(cv2, "IMWRITE_EXR_COMPRESSION_ZIPS", 2),
                    "zip": getattr(cv2, "IMWRITE_EXR_COMPRESSION_ZIP", 3),
                    "piz": getattr(cv2, "IMWRITE_EXR_COMPRESSION_PIZ", 4),
                }
                comp_val = comp_map.get(compression, comp_map["zip"])
                params.append(comp_key)
                params.append(comp_val)
            success = cv2.imwrite(filepath, bgr, params)
            if success:
                return
        except Exception:
            pass

    # 2) imageio: ensure FreeImage DLL is available then write by .exr extension
    if not _HAS_IMAGEIO:
        raise RuntimeError(
            "SaveImageEXR: No EXR backend available. Install imageio and FreeImage: "
            "pip install imageio imageio[freeimage]"
        )
    try:
        # Ensure FreeImage binary is present (downloads on first use if needed)
        getattr(imageio.plugins.freeimage, "download", lambda: None)()
    except Exception:
        pass
    try:
        # Write by extension so imageio picks the EXR/FreeImage plugin
        imageio.imwrite(filepath, data_f32 if data.dtype != "float16" else data)
        return
    except (TypeError, Exception):
        pass
    raise RuntimeError(
        "Could not write EXR. Install: pip install imageio imageio[freeimage] "
        "(downloads FreeImage on first EXR save)."
    )


def _get_exr_bit_depth_from_file(filepath: str) -> int:
    """
    Read EXR file header and return the channel pixel type: 16 for HALF, 32 for FLOAT.
    OpenEXR chlist: pixeltype 0=UINT, 1=HALF, 2=FLOAT. Returns 16 or 32 (bytes per channel).
    """
    import struct
    with open(filepath, "rb") as f:
        magic = struct.unpack("<I", f.read(4))[0]
        if magic != 20000630:
            raise ValueError(f"Not an EXR file (magic={magic})")
        f.read(4)  # version
        while True:
            name = _read_null_term_string(f)
            if not name:
                break
            atype = _read_null_term_string(f)
            (size,) = struct.unpack("<i", f.read(4))
            value = f.read(size)
            if name == "channels" and atype == "chlist":
                # chlist: for each channel: name (null-term), pixeltype (int), pLinear (1), reserved (3), xSampling (int), ySampling (int)
                idx = 0
                while idx < len(value) and value[idx] != 0:
                    idx += 1
                idx += 1  # skip null
                if idx + 4 <= len(value):
                    (pixeltype,) = struct.unpack("<i", value[idx : idx + 4])
                    # 0=UINT, 1=HALF, 2=FLOAT
                    if pixeltype == 1:
                        return 16
                    if pixeltype in (0, 2):
                        return 32
                    return 32  # unknown, treat as 32
                break
    return 32  # default if not found


def _read_null_term_string(f) -> str:
    buf = []
    while True:
        b = f.read(1)
        if not b or b == b"\x00":
            break
        buf.append(b)
    return b"".join(buf).decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Prepare Image for High Bit Depth (optional linear + deband)
# ---------------------------------------------------------------------------
class PrepareImageHighBitDepth(IO.ComfyNode):
    """
    Prepares IMAGE for high bit-depth export: optional sRGB→linear and debanding.
    Use before SaveImageEXR or when feeding frames for ProRes HQ.
    """

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="PrepareImageHighBitDepth",
            display_name="Prepare Image High Bit Depth",
            category="image/processing",
            description="Optional sRGB→linear and debanding for HDR/ProRes export.",
            inputs=[
                IO.Image.Input("image", tooltip="Image in 0–1 range (e.g. from VAE Decode)"),
                IO.Boolean.Input("output_linear", default=False, tooltip="Convert sRGB to linear for EXR/VFX"),
                IO.Float.Input(
                    "deband_strength",
                    default=0.0,
                    min=0.0,
                    max=2.0,
                    step=0.05,
                    tooltip="Bilateral deband strength (0=off). Requires OpenCV.",
                ),
            ],
            outputs=[IO.Image.Output(display_name="image")],
        )

    @classmethod
    def execute(cls, image, output_linear: bool, deband_strength: float) -> IO.NodeOutput:
        out = image.clone()
        if output_linear:
            out = _srgb_to_linear(out)
        if deband_strength > 0 and out.shape[-1] >= 3:
            # Per-frame for batch
            for i in range(out.shape[0]):
                out[i] = _deband_image_numpy(out[i], deband_strength)
        return IO.NodeOutput(out)


# ---------------------------------------------------------------------------
# Save Image EXR (16 or 32 bit)
# ---------------------------------------------------------------------------
class SaveImageEXR(IO.ComfyNode):
    """Saves a single image as EXR (16 or 32 bit). Preserves values > 1.0 if present."""

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="SaveImageEXR",
            display_name="Save Image EXR",
            category="image/save",
            description="Save image as 16- or 32-bit EXR for HDR/VFX.",
            inputs=[
                IO.Image.Input("image", tooltip="Image (0–1 or HDR > 1)"),
                IO.String.Input("filename_prefix", default="image/ComfyUI_EXR", tooltip="Filename prefix"),
                IO.Combo.Input(
                    "bit_depth",
                    options=["16", "32"],
                    default="32",
                    tooltip="EXR bit depth",
                ),
                IO.Combo.Input(
                    "compression",
                    options=["none", "zip", "rle", "zips", "piz"],
                    default="zip",
                    tooltip="EXR compression",
                ),
            ],
            outputs=[IO.Image.Output(display_name="image", tooltip="Pass-through image for chaining")],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, image, filename_prefix: str, bit_depth: str, compression: str) -> IO.NodeOutput:
        if not _HAS_CV2 and not _HAS_IMAGEIO:
            raise RuntimeError(
                "SaveImageEXR requires opencv-python and/or imageio. "
                "Install: pip install opencv-python  (recommended for EXR)"
            )
        img = image[0]  # first batch item
        h, w = img.shape[0], img.shape[1]
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), w, h
        )
        filepath = os.path.join(full_output_folder, f"{filename}_{counter:05}_.exr")
        os.makedirs(full_output_folder, exist_ok=True)
        # EXR: (H,W,C) float32; values can be > 1
        data = img.clamp(0.0, None).float().cpu().numpy()
        if bit_depth == "16":
            data = data.astype("float16")
        comp_map = {"none": "none", "zip": "zip", "rle": "rle", "zips": "zips", "piz": "piz"}
        comp_val = comp_map.get(compression, "zip")
        try:
            _write_exr(filepath, data, comp_val)
        except Exception as e:
            raise RuntimeError(
                f"SaveImageEXR failed: {e}. "
                "Install opencv-python for EXR (recommended): pip install opencv-python. "
                "Or imageio with FreeImage: pip install imageio imageio[freeimage]."
            ) from e
        return IO.NodeOutput(image, ui=UI.PreviewImage(image))


# ---------------------------------------------------------------------------
# Save Video as EXR sequence (16 or 32 bit)
# ---------------------------------------------------------------------------
class SaveVideoEXRSequence(IO.ComfyNode):
    """Saves video frames as an EXR sequence (16 or 32 bit). For 4K ProRes later: ffmpeg -i 'seq_%06d.exr' ..."""

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="SaveVideoEXRSequence",
            display_name="Save Video EXR Sequence",
            category="image/video",
            description="Save video as 16- or 32-bit EXR sequence for HDR/VFX or ProRes encoding.",
            inputs=[
                IO.Video.Input("video", tooltip="Video to save as EXR sequence"),
                IO.String.Input("filename_prefix", default="video/ComfyUI_EXR", tooltip="Prefix for frame files"),
                IO.Combo.Input("bit_depth", options=["16", "32"], default="32"),
                IO.Combo.Input("compression", options=["none", "zip", "rle", "zips", "piz"], default="zip"),
            ],
            outputs=[IO.Video.Output(display_name="video", tooltip="Pass-through video for chaining")],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, video, filename_prefix: str, bit_depth: str, compression: str) -> IO.NodeOutput:
        if not _HAS_IMAGEIO:
            raise RuntimeError("SaveVideoEXRSequence requires 'imageio'. Install: pip install imageio")
        components = video.get_components()
        images = components.images  # [B, H, W, C]
        h, w = images.shape[1], images.shape[2]
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), w, h
        )
        os.makedirs(full_output_folder, exist_ok=True)
        comp_map = {"none": "none", "zip": "zip", "rle": "rle", "zips": "zips", "piz": "piz"}
        comp_val = comp_map.get(compression, "zip")
        for i in range(images.shape[0]):
            frame = images[i].clamp(0.0, None).float().cpu().numpy()
            if bit_depth == "16":
                frame = frame.astype("float16")
            path = os.path.join(full_output_folder, f"{filename}_{counter:05}_{i:06d}.exr")
            _write_exr(path, frame, comp_val)
        first_name = f"{filename}_{counter:05}_000000.exr"
        return IO.NodeOutput(
            video,
            ui=UI.PreviewVideo([UI.SavedResult(first_name, subfolder, IO.FolderType.output)]),
        )


# ---------------------------------------------------------------------------
# Save Video ProRes HQ (4K ProRes 4444 XQ with optional linear + deband)
# ---------------------------------------------------------------------------
class SaveVideoProResHQ(IO.ComfyNode):
    """
    Saves video as ProRes 4444 XQ (12-bit) with optional linear→bt709 and debanding (gradfun).
    Uses ffmpeg when available for best quality; otherwise falls back to PyAV 10-bit.
    """

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="SaveVideoProResHQ",
            display_name="Save Video ProRes HQ (4K)",
            category="image/video",
            description="ProRes 4444 XQ with optional linear/deband pipeline. Needs ffmpeg for gradfun.",
            inputs=[
                IO.Video.Input("video", tooltip="Video to save"),
                IO.String.Input("filename_prefix", default="video/ComfyUI_ProResHQ", tooltip="Filename prefix"),
                IO.Boolean.Input(
                    "apply_deband",
                    default=True,
                    tooltip="Apply gradfun-style debanding (requires ffmpeg)",
                ),
                IO.Boolean.Input(
                    "linear_to_bt709",
                    default=True,
                    tooltip="Process in linear then convert to bt709 for ProRes",
                ),
            ],
            outputs=[IO.Video.Output(display_name="video", tooltip="Pass-through video for chaining")],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        video,
        filename_prefix: str,
        apply_deband: bool,
        linear_to_bt709: bool,
    ) -> IO.NodeOutput:
        import av
        components = video.get_components()
        images = components.images
        frame_rate = components.frame_rate
        audio = components.audio
        h, w = images.shape[1], images.shape[2]
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), w, h
        )
        os.makedirs(full_output_folder, exist_ok=True)
        mov_path = os.path.join(full_output_folder, f"{filename}_{counter:05}_.mov")
        ffmpeg_path = _find_ffmpeg()

        if ffmpeg_path and (apply_deband or linear_to_bt709) and _HAS_IMAGEIO:
            # Pipeline: write temp EXR sequence → ffmpeg (format=gbrpf32le, zscale, gradfun, prores_ks)
            with tempfile.TemporaryDirectory(prefix="comfy_prores_") as tmpdir:
                for i in range(images.shape[0]):
                    frame = images[i].clamp(0.0, 1.0).float().cpu().numpy()
                    path = os.path.join(tmpdir, f"frame_{i:06d}.exr")
                    _write_exr(path, frame, "zip")
                # Build filter: format=gbrpf32le (float RGB) → optional linear → gradfun → bt709 → yuv444p10le
                vf_parts = ["format=gbrpf32le"]
                if linear_to_bt709:
                    vf_parts.append("zscale=t=linear")
                if apply_deband:
                    vf_parts.append("gradfun=20:16")
                if linear_to_bt709:
                    vf_parts.append("zscale=t=bt709")
                vf_parts.append("format=yuv444p10le")
                vf = ",".join(vf_parts)
                input_pattern = os.path.join(tmpdir, "frame_%06d.exr").replace("\\", "/")
                cmd = [
                    ffmpeg_path,
                    "-y",
                    "-framerate", str(frame_rate),
                    "-i", input_pattern,
                    "-vf", vf,
                    "-c:v", "prores_ks",
                    "-profile:v", "5",
                    "-pix_fmt", "yuv444p10le",
                ]
                if audio is not None:
                    # Write temp WAV for audio then mux (simplest)
                    rate = int(audio["sample_rate"])
                    num_frames = images.shape[0]
                    need = math.ceil((rate / frame_rate) * num_frames)
                    wav = audio["waveform"]
                    if wav.shape[2] > need:
                        wav = wav[:, :, :need]
                    elif wav.shape[2] < need:
                        pad = need - wav.shape[2]
                        wav = torch.cat([
                            wav,
                            torch.zeros(wav.shape[0], wav.shape[1], pad, device=wav.device, dtype=wav.dtype),
                        ], dim=2)
                    wav_np = wav.movedim(2, 1).reshape(-1).float().cpu().numpy()
                    wav_path = os.path.join(tmpdir, "audio.wav")
                    import wave
                    with wave.open(wav_path, "wb") as f:
                        f.setnchannels(wav.shape[1])
                        f.setsampwidth(2)
                        f.setframerate(rate)
                        f.writeframes((wav_np * 32767).clip(-32768, 32767).astype("int16").tobytes())
                    cmd.extend(["-i", wav_path, "-c:a", "pcm_s16le", "-shortest"])
                cmd.append(mov_path)
                try:
                    subprocess.run(cmd, check=True, capture_output=True)
                except subprocess.CalledProcessError as e:
                    raise RuntimeError(f"ffmpeg failed: {e.stderr.decode(errors='replace')}")
        else:
            # Fallback: PyAV ProRes 4444 (no gradfun/zscale)
            frame_rate_frac = __import__("fractions").Fraction(round(frame_rate * 1000), 1000)
            with av.open(mov_path, mode="w", format="mov") as out:
                stream = out.add_stream("prores_4444", rate=frame_rate_frac)
                stream.width = w
                stream.height = h
                stream.pix_fmt = "yuv444p10le"
                for i in range(images.shape[0]):
                    frame = images[i].clamp(0.0, 1.0)
                    img = (frame * 65535).clamp(0, 65535).short().cpu().numpy()
                    av_frame = av.VideoFrame.from_ndarray(img, format="rgb48le")
                    av_frame = av_frame.reformat(format="yuv444p10le")
                    for pkt in stream.encode(av_frame):
                        out.mux(pkt)
                for pkt in stream.encode(None):
                    out.mux(pkt)
                if audio is not None:
                    rate = int(audio["sample_rate"])
                    num_frames = images.shape[0]
                    need = math.ceil((rate / frame_rate) * num_frames)
                    wav = audio["waveform"]
                    if wav.shape[2] > need:
                        wav = wav[:, :, :need]
                    elif wav.shape[2] < need:
                        pad = need - wav.shape[2]
                        wav = torch.cat([
                            wav,
                            torch.zeros(wav.shape[0], wav.shape[1], pad, device=wav.device, dtype=wav.dtype),
                        ], dim=2)
                    audio_stream = out.add_stream("pcm_s16le", rate=rate)
                    audio_np = wav.movedim(2, 1).reshape(1, -1).float().cpu().numpy()
                    af = av.AudioFrame.from_ndarray(audio_np, format="flt", layout="mono" if wav.shape[1] == 1 else "stereo")
                    af.sample_rate = rate
                    af.pts = 0
                    for pkt in audio_stream.encode(af):
                        out.mux(pkt)
                    for pkt in audio_stream.encode(None):
                        out.mux(pkt)

        return IO.NodeOutput(
            video,
            ui=UI.PreviewVideo([UI.SavedResult(os.path.basename(mov_path), subfolder, IO.FolderType.output)]),
        )


# ---------------------------------------------------------------------------
# Extension
# ---------------------------------------------------------------------------
class HighBitExportExtension(ComfyExtension):
    async def get_node_list(self) -> list:
        return [
            PrepareImageHighBitDepth,
            SaveImageEXR,
            SaveVideoEXRSequence,
            SaveVideoProResHQ,
        ]


async def comfy_entrypoint() -> HighBitExportExtension:
    return HighBitExportExtension()
