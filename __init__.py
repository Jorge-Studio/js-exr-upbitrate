# High bit-depth export nodes: 16/32-bit EXR, ProRes 4444 XQ
# Converted to traditional NODE_CLASS_MAPPINGS for compatibility

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
    """Simple debanding: bilateral filter. Returns tensor on same device as input."""
    if not _HAS_CV2 or strength <= 0:
        return img_t
    img = np.ascontiguousarray(img_t.clamp(0.0, 1.0).cpu().numpy().astype("float32"))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    sigma = max(1.0, 20.0 * strength)
    filtered = cv2.bilateralFilter(img_bgr, 5, sigma, sigma)
    out_rgb = cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)
    out = torch.from_numpy(out_rgb)
    return out.to(device=img_t.device, dtype=img_t.dtype)


def _find_ffmpeg() -> Optional[str]:
    return shutil.which("ffmpeg")


def _write_exr(filepath: str, data: "np.ndarray", compression: str = "zip") -> None:
    """Write float32/float16 array as EXR."""
    if data.dtype == "float16":
        data_f32 = data.astype("float32")
        request_16bit = True
    else:
        data_f32 = np.ascontiguousarray(data.astype("float32"))
        request_16bit = False

    # OpenCV: most reliable for EXR
    if _HAS_CV2:
        try:
            bgr = cv2.cvtColor(data_f32, cv2.COLOR_RGB2BGR)
            params = []
            exr_type_key = getattr(cv2, "IMWRITE_EXR_TYPE", None)
            if exr_type_key is not None:
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

    # imageio fallback
    if not _HAS_IMAGEIO:
        raise RuntimeError(
            "SaveImageEXR: No EXR backend available. Install opencv-python or imageio."
        )
    try:
        imageio.imwrite(filepath, data_f32 if data.dtype != "float16" else data)
        return
    except Exception:
        pass
    raise RuntimeError("Could not write EXR. Install opencv-python for EXR support.")


# ---------------------------------------------------------------------------
# Prepare Image for High Bit Depth
# ---------------------------------------------------------------------------
class PrepareImageHighBitDepth:
    """Prepares IMAGE for high bit-depth export: optional sRGB→linear and debanding."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "output_linear": ("BOOLEAN", {"default": False, 
                    "tooltip": "Convert sRGB to linear for EXR/VFX"}),
                "deband_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Bilateral deband strength (0=off). Requires OpenCV."}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "image/processing"
    DESCRIPTION = "Optional sRGB→linear and debanding for HDR/ProRes export."
    
    def execute(self, image, output_linear: bool, deband_strength: float):
        out = image.clone()
        if output_linear:
            out = _srgb_to_linear(out)
        if deband_strength > 0 and out.shape[-1] >= 3:
            for i in range(out.shape[0]):
                out[i] = _deband_image_numpy(out[i], deband_strength)
        return (out,)


# ---------------------------------------------------------------------------
# Save Image EXR (16 or 32 bit)
# ---------------------------------------------------------------------------
class SaveImageEXR:
    """Saves a single image as EXR (16 or 32 bit). Preserves values > 1.0 if present."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "ComfyUI_EXR"}),
                "bit_depth": (["16", "32"], {"default": "32"}),
                "compression": (["none", "zip", "rle", "zips", "piz"], {"default": "zip"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "image/save"
    OUTPUT_NODE = True
    DESCRIPTION = "Save image as 16- or 32-bit EXR for HDR/VFX."
    
    def execute(self, image, filename_prefix: str, bit_depth: str, compression: str):
        if not _HAS_CV2 and not _HAS_IMAGEIO:
            raise RuntimeError(
                "SaveImageEXR requires opencv-python and/or imageio. "
                "Install: pip install opencv-python"
            )
        img = image[0]
        h, w = img.shape[0], img.shape[1]
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), w, h
        )
        filepath = os.path.join(full_output_folder, f"{filename}_{counter:05d}.exr")
        os.makedirs(full_output_folder, exist_ok=True)
        
        data = img.clamp(0.0, None).float().cpu().numpy()
        if bit_depth == "16":
            data = data.astype("float16")
        
        _write_exr(filepath, data, compression)
        
        return {"ui": {"images": [{"filename": os.path.basename(filepath), "subfolder": subfolder, "type": "output"}]}, 
                "result": (image,)}


# ---------------------------------------------------------------------------
# Save Video as EXR sequence (16 or 32 bit)
# ---------------------------------------------------------------------------
class SaveVideoEXRSequence:
    """Saves image batch as an EXR sequence (16 or 32 bit)."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "ComfyUI_EXR_seq"}),
                "bit_depth": (["16", "32"], {"default": "32"}),
                "compression": (["none", "zip", "rle", "zips", "piz"], {"default": "zip"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "execute"
    CATEGORY = "image/video"
    OUTPUT_NODE = True
    DESCRIPTION = "Save image batch as 16- or 32-bit EXR sequence for HDR/VFX."
    
    def execute(self, images, filename_prefix: str, bit_depth: str, compression: str):
        if not _HAS_CV2 and not _HAS_IMAGEIO:
            raise RuntimeError("SaveVideoEXRSequence requires opencv-python or imageio.")
        
        h, w = images.shape[1], images.shape[2]
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), w, h
        )
        os.makedirs(full_output_folder, exist_ok=True)
        
        results = []
        for i in range(images.shape[0]):
            frame = images[i].clamp(0.0, None).float().cpu().numpy()
            if bit_depth == "16":
                frame = frame.astype("float16")
            path = os.path.join(full_output_folder, f"{filename}_{counter:05d}_{i:06d}.exr")
            _write_exr(path, frame, compression)
            results.append({"filename": os.path.basename(path), "subfolder": subfolder, "type": "output"})
        
        return {"ui": {"images": results}, "result": (images,)}


NODE_CLASS_MAPPINGS = {
    "PrepareImageHighBitDepth": PrepareImageHighBitDepth,
    "SaveImageEXR": SaveImageEXR,
    "SaveVideoEXRSequence": SaveVideoEXRSequence,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PrepareImageHighBitDepth": "Prepare Image High Bit Depth",
    "SaveImageEXR": "Save Image EXR",
    "SaveVideoEXRSequence": "Save Video EXR Sequence",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
