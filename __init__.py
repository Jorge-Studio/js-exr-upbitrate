# High bit-depth export nodes: 16/32-bit EXR with Log format support
# Version 2.0 - Improved color space handling and tonal precision

import os
import sys
import math
import tempfile
import subprocess
import shutil
import json
from typing import Optional, Tuple, List

import numpy as np
import torch
import folder_paths

# Optional dependencies
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

try:
    import OpenEXR
    import Imath
    _HAS_OPENEXR = True
except ImportError:
    _HAS_OPENEXR = False


# =============================================================================
# COLOR SPACE TRANSFORMS - High precision implementations
# =============================================================================

def _srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    """Convert sRGB to linear RGB. Preserves values > 1.0 for HDR headroom."""
    # Use double precision for the conversion to maximize tonal precision
    x_double = x.double()
    out = torch.where(
        x_double <= 0.04045,
        x_double / 12.92,
        torch.pow((x_double + 0.055) / 1.055, 2.4)
    )
    return out.float()  # Return as float32 for compatibility


def _linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    """Convert linear RGB to sRGB. Preserves values > 1.0."""
    x_double = x.double().clamp(min=0.0)
    out = torch.where(
        x_double <= 0.0031308,
        x_double * 12.92,
        1.055 * torch.pow(x_double, 1.0 / 2.4) - 0.055
    )
    return out.float()


# -----------------------------------------------------------------------------
# LOG FORMAT TRANSFORMS - Industry standard log curves
# -----------------------------------------------------------------------------

def _linear_to_log_c(x: torch.Tensor) -> torch.Tensor:
    """Convert linear to ARRI LogC3 (EI 800). Standard for ARRI cameras."""
    # ARRI LogC3 constants (EI 800)
    cut = 0.010591
    a = 5.555556
    b = 0.052272
    c = 0.247190
    d = 0.385537
    e = 5.367655
    f = 0.092809
    
    x_double = x.double().clamp(min=0.0)
    out = torch.where(
        x_double > cut,
        c * torch.log10(a * x_double + b) + d,
        e * x_double + f
    )
    return out.float()


def _log_c_to_linear(x: torch.Tensor) -> torch.Tensor:
    """Convert ARRI LogC3 to linear."""
    cut = 0.010591
    a = 5.555556
    b = 0.052272
    c = 0.247190
    d = 0.385537
    e = 5.367655
    f = 0.092809
    
    cut_log = e * cut + f
    
    x_double = x.double()
    out = torch.where(
        x_double > cut_log,
        (torch.pow(10.0, (x_double - d) / c) - b) / a,
        (x_double - f) / e
    )
    return out.float().clamp(min=0.0)


def _linear_to_slog3(x: torch.Tensor) -> torch.Tensor:
    """Convert linear to Sony S-Log3."""
    x_double = x.double().clamp(min=0.0)
    out = torch.where(
        x_double >= 0.01125000,
        (420.0 + torch.log10((x_double + 0.01) / (0.18 + 0.01)) * 261.5) / 1023.0,
        (x_double * (171.2102946929 - 95.0) / 0.01125000 + 95.0) / 1023.0
    )
    return out.float()


def _slog3_to_linear(x: torch.Tensor) -> torch.Tensor:
    """Convert Sony S-Log3 to linear."""
    x_double = x.double() * 1023.0
    out = torch.where(
        x_double >= 171.2102946929,
        torch.pow(10.0, (x_double - 420.0) / 261.5) * (0.18 + 0.01) - 0.01,
        (x_double - 95.0) * 0.01125000 / (171.2102946929 - 95.0)
    )
    return out.float().clamp(min=0.0)


def _linear_to_vlog(x: torch.Tensor) -> torch.Tensor:
    """Convert linear to Panasonic V-Log."""
    cut = 0.01
    b = 0.00873
    c = 0.241514
    d = 0.598206
    
    x_double = x.double().clamp(min=0.0)
    out = torch.where(
        x_double >= cut,
        c * torch.log10(x_double + b) + d,
        5.6 * x_double + 0.125
    )
    return out.float()


def _vlog_to_linear(x: torch.Tensor) -> torch.Tensor:
    """Convert Panasonic V-Log to linear."""
    cut_log = 0.181
    b = 0.00873
    c = 0.241514
    d = 0.598206
    
    x_double = x.double()
    out = torch.where(
        x_double >= cut_log,
        torch.pow(10.0, (x_double - d) / c) - b,
        (x_double - 0.125) / 5.6
    )
    return out.float().clamp(min=0.0)


def _linear_to_clog3(x: torch.Tensor) -> torch.Tensor:
    """Convert linear to Canon Log 3."""
    x_double = x.double().clamp(min=0.0)
    
    # Canon Log 3 curve
    out = torch.where(
        x_double >= 0.014,
        0.42889912 * torch.log10(x_double * 14.98325 + 1.0) + 0.069886632,
        -0.42889912 * torch.log10(-x_double * 14.98325 + 1.0) + 0.069886632
    )
    return out.float()


def _linear_to_redlog3g10(x: torch.Tensor) -> torch.Tensor:
    """Convert linear to RED Log3G10."""
    a = 0.224282
    b = 155.975327
    c = 0.01
    g = 15.1927
    
    x_double = x.double().clamp(min=-0.01)
    
    out = torch.where(
        x_double < 0.0,
        x_double * g,
        a * torch.log10((x_double + c) * b + 1.0)
    )
    return out.float()


def _linear_to_davinci_intermediate(x: torch.Tensor) -> torch.Tensor:
    """Convert linear to DaVinci Intermediate (Wide Gamut)."""
    a = 0.0075
    b = 7.0
    c = 0.07329248
    m = 10.44426855
    lin_cut = 0.00262409
    log_cut = 0.02740668
    
    x_double = x.double().clamp(min=0.0)
    
    out = torch.where(
        x_double <= lin_cut,
        x_double * m,
        (torch.log2(x_double + a) + b) * c
    )
    return out.float()


# Color space conversion dispatcher
COLOR_SPACE_TRANSFORMS = {
    "Linear": (lambda x: x, lambda x: x),  # Identity
    "sRGB": (_linear_to_srgb, _srgb_to_linear),
    "ARRI LogC3": (_linear_to_log_c, _log_c_to_linear),
    "Sony S-Log3": (_linear_to_slog3, _slog3_to_linear),
    "Panasonic V-Log": (_linear_to_vlog, _vlog_to_linear),
    "Canon Log 3": (_linear_to_clog3, None),
    "RED Log3G10": (_linear_to_redlog3g10, None),
    "DaVinci Intermediate": (_linear_to_davinci_intermediate, None),
}

LOG_FORMATS = ["Linear", "ARRI LogC3", "Sony S-Log3", "Panasonic V-Log", 
               "Canon Log 3", "RED Log3G10", "DaVinci Intermediate"]

INPUT_COLOR_SPACES = ["sRGB (ComfyUI Default)", "Linear", "ARRI LogC3", 
                      "Sony S-Log3", "Panasonic V-Log"]


# =============================================================================
# IMAGE PROCESSING UTILITIES
# =============================================================================

def _apply_exposure(x: torch.Tensor, stops: float) -> torch.Tensor:
    """Apply exposure adjustment in stops (linear multiply)."""
    if stops == 0.0:
        return x
    multiplier = math.pow(2.0, stops)
    return x * multiplier


def _apply_contrast(x: torch.Tensor, contrast: float, pivot: float = 0.18) -> torch.Tensor:
    """Apply contrast around a pivot point (default 18% grey)."""
    if contrast == 1.0:
        return x
    # Work in log space for smoother contrast
    x_ratio = x / pivot
    x_contrast = torch.pow(x_ratio.clamp(min=1e-10), contrast) * pivot
    return x_contrast


def _apply_lift_gamma_gain(x: torch.Tensor, lift: float, gamma: float, gain: float) -> torch.Tensor:
    """Apply lift/gamma/gain color correction."""
    # Lift (shadows) - add offset
    out = x + lift
    # Gain (highlights) - multiply
    out = out * gain
    # Gamma (midtones) - power curve
    if gamma != 1.0:
        out = torch.pow(out.clamp(min=0.0), 1.0 / gamma)
    return out


def _apply_curves(x: torch.Tensor, curve_points: List[Tuple[float, float]]) -> torch.Tensor:
    """Apply a custom curve defined by control points.
    Points should be list of (input, output) tuples from 0.0 to 1.0.
    """
    if not curve_points or len(curve_points) < 2:
        return x
    
    # Sort points by input value
    points = sorted(curve_points, key=lambda p: p[0])
    
    # Build lookup table for efficiency
    lut_size = 4096  # High precision LUT
    lut = torch.zeros(lut_size, dtype=torch.float32, device=x.device)
    
    for i in range(lut_size):
        input_val = i / (lut_size - 1)
        
        # Find surrounding control points
        lower_idx = 0
        for j, (px, py) in enumerate(points):
            if px <= input_val:
                lower_idx = j
            else:
                break
        
        upper_idx = min(lower_idx + 1, len(points) - 1)
        
        # Interpolate
        x1, y1 = points[lower_idx]
        x2, y2 = points[upper_idx]
        
        if x2 == x1:
            lut[i] = y1
        else:
            t = (input_val - x1) / (x2 - x1)
            # Smooth interpolation (cubic)
            t = t * t * (3.0 - 2.0 * t)
            lut[i] = y1 + t * (y2 - y1)
    
    # Apply LUT
    x_normalized = (x.clamp(0.0, 1.0) * (lut_size - 1)).long()
    return lut[x_normalized]


def _deband_image(img_t: torch.Tensor, strength: float) -> torch.Tensor:
    """Improved debanding using dithering and bilateral filter."""
    if strength <= 0:
        return img_t
    
    # Add subtle noise to break up banding (blue noise would be ideal)
    noise_amount = strength * 0.002  # Very subtle
    noise = torch.randn_like(img_t) * noise_amount
    out = img_t + noise
    
    # Apply bilateral filter if OpenCV available
    if _HAS_CV2 and strength > 0.5:
        for i in range(out.shape[0]):
            frame = out[i].cpu().numpy().astype("float32")
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            sigma = max(1.0, 15.0 * strength)
            filtered = cv2.bilateralFilter(frame_bgr, 5, sigma, sigma)
            out_rgb = cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)
            out[i] = torch.from_numpy(out_rgb).to(device=img_t.device, dtype=img_t.dtype)
    
    return out


def _find_ffmpeg() -> Optional[str]:
    return shutil.which("ffmpeg")


# =============================================================================
# EXR WRITING - Optimized for maximum precision
# =============================================================================

def _write_exr(filepath: str, data: np.ndarray, compression: str = "zip", 
               bit_depth: str = "32", color_space: str = "Linear") -> None:
    """Write EXR with proper bit depth and color space metadata."""
    
    # CRITICAL: Do NOT clamp data - preserve full range including values > 1.0
    # This maintains highlight headroom for grading
    data_f32 = np.ascontiguousarray(data.astype("float32"))
    request_16bit = (bit_depth == "16")
    
    if _HAS_OPENEXR:
        try:
            h, w, c = data_f32.shape
            header = OpenEXR.Header(w, h)
            
            # Add color space metadata (for software that reads it)
            # Using standard ACES/OpenColorIO attribute names
            header['chromaticities'] = Imath.Chromaticities(
                Imath.V2f(0.64, 0.33),   # red
                Imath.V2f(0.30, 0.60),   # green
                Imath.V2f(0.15, 0.06),   # blue
                Imath.V2f(0.3127, 0.3290) # white (D65)
            )
            
            # Pixel type for channels
            if request_16bit:
                pixel_type = Imath.PixelType(Imath.PixelType.HALF)
                r_data = data_f32[:, :, 0].astype('float16').tobytes()
                g_data = data_f32[:, :, 1].astype('float16').tobytes()
                b_data = data_f32[:, :, 2].astype('float16').tobytes()
            else:
                pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
                r_data = data_f32[:, :, 0].tobytes()
                g_data = data_f32[:, :, 1].tobytes()
                b_data = data_f32[:, :, 2].tobytes()
            
            header['channels'] = {
                'R': Imath.Channel(pixel_type),
                'G': Imath.Channel(pixel_type),
                'B': Imath.Channel(pixel_type),
            }
            
            out = OpenEXR.OutputFile(filepath, header)
            out.writePixels({'R': r_data, 'G': g_data, 'B': b_data})
            out.close()
            return
        except Exception as e:
            print(f"OpenEXR write failed: {e}, trying fallback...")
    
    # OpenCV fallback
    if _HAS_CV2:
        try:
            bgr = cv2.cvtColor(data_f32, cv2.COLOR_RGB2BGR)
            params = []
            exr_type_key = getattr(cv2, "IMWRITE_EXR_TYPE", None)
            if exr_type_key is not None:
                exr_half = getattr(cv2, "IMWRITE_EXR_TYPE_HALF", 1)
                exr_float = getattr(cv2, "IMWRITE_EXR_TYPE_FLOAT", 2)
                params.extend([exr_type_key, exr_half if request_16bit else exr_float])
            comp_key = getattr(cv2, "IMWRITE_EXR_COMPRESSION", None)
            if comp_key is not None:
                comp_map = {
                    "none": getattr(cv2, "IMWRITE_EXR_COMPRESSION_NO", 0),
                    "rle": getattr(cv2, "IMWRITE_EXR_COMPRESSION_RLE", 1),
                    "zips": getattr(cv2, "IMWRITE_EXR_COMPRESSION_ZIPS", 2),
                    "zip": getattr(cv2, "IMWRITE_EXR_COMPRESSION_ZIP", 3),
                    "piz": getattr(cv2, "IMWRITE_EXR_COMPRESSION_PIZ", 4),
                    "dwaa": getattr(cv2, "IMWRITE_EXR_COMPRESSION_DWAA", 6),
                }
                params.extend([comp_key, comp_map.get(compression, comp_map["zip"])])
            if cv2.imwrite(filepath, bgr, params):
                return
        except Exception:
            pass

    if _HAS_IMAGEIO:
        try:
            imageio.imwrite(filepath, data_f32)
            return
        except Exception:
            pass
    
    raise RuntimeError("Could not write EXR. Install: pip install openexr imath")


# =============================================================================
# NODE: Color Space Converter
# =============================================================================
class ColorSpaceConverter:
    """Convert between color spaces with high precision."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "input_space": (INPUT_COLOR_SPACES, {"default": "sRGB (ComfyUI Default)"}),
                "output_space": (["Linear", "sRGB"] + LOG_FORMATS[1:], {"default": "Linear"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "image/color"
    DESCRIPTION = "Convert between color spaces (sRGB, Linear, Log formats)."
    
    def execute(self, image, input_space: str, output_space: str):
        out = image.clone()
        
        # Step 1: Convert input to linear
        if "sRGB" in input_space:
            out = _srgb_to_linear(out)
        elif input_space == "ARRI LogC3":
            out = _log_c_to_linear(out)
        elif input_space == "Sony S-Log3":
            out = _slog3_to_linear(out)
        elif input_space == "Panasonic V-Log":
            out = _vlog_to_linear(out)
        # Linear stays as-is
        
        # Step 2: Convert linear to output space
        if output_space == "sRGB":
            out = _linear_to_srgb(out)
        elif output_space == "ARRI LogC3":
            out = _linear_to_log_c(out)
        elif output_space == "Sony S-Log3":
            out = _linear_to_slog3(out)
        elif output_space == "Panasonic V-Log":
            out = _linear_to_vlog(out)
        elif output_space == "Canon Log 3":
            out = _linear_to_clog3(out)
        elif output_space == "RED Log3G10":
            out = _linear_to_redlog3g10(out)
        elif output_space == "DaVinci Intermediate":
            out = _linear_to_davinci_intermediate(out)
        # Linear stays as-is
        
        return (out,)


# =============================================================================
# NODE: Color Grading Controller
# =============================================================================
class ColorGradingController:
    """Professional color grading with exposure, contrast, lift/gamma/gain."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "exposure": ("FLOAT", {"default": 0.0, "min": -5.0, "max": 5.0, "step": 0.1,
                    "tooltip": "Exposure adjustment in stops"}),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05,
                    "tooltip": "Contrast around 18% grey"}),
                "lift": ("FLOAT", {"default": 0.0, "min": -0.5, "max": 0.5, "step": 0.01,
                    "tooltip": "Shadow lift (offset)"}),
                "gamma": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05,
                    "tooltip": "Midtone gamma"}),
                "gain": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05,
                    "tooltip": "Highlight gain (multiply)"}),
                "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            },
            "optional": {
                "input_is_linear": ("BOOLEAN", {"default": True,
                    "tooltip": "Enable if input is from PrepareImageHighBitDepth (already linear). Disable for raw ComfyUI images."}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "image/color"
    DESCRIPTION = "Professional color grading: exposure, contrast, lift/gamma/gain. Works in linear space."
    
    def execute(self, image, exposure: float, contrast: float, 
                lift: float, gamma: float, gain: float, saturation: float,
                input_is_linear: bool = True):
        # Early return if all parameters are neutral - preserves full precision
        if (input_is_linear and exposure == 0.0 and contrast == 1.0 and 
            lift == 0.0 and gamma == 1.0 and gain == 1.0 and saturation == 1.0):
            return (image,)
        
        out = image.clone()
        
        # Only convert to linear if input is NOT already linear
        if not input_is_linear:
            out = _srgb_to_linear(out)
        
        # Apply grading operations (all in linear space)
        out = _apply_exposure(out, exposure)
        out = _apply_contrast(out, contrast)
        out = _apply_lift_gamma_gain(out, lift, gamma, gain)
        
        # Saturation (in linear space)
        if saturation != 1.0:
            luminance = 0.2126 * out[..., 0] + 0.7152 * out[..., 1] + 0.0722 * out[..., 2]
            luminance = luminance.unsqueeze(-1)
            out = luminance + saturation * (out - luminance)
        
        return (out,)


# =============================================================================
# NODE: HDR Curve Editor
# =============================================================================
class HDRCurveEditor:
    """Apply custom tone curve with HDR headroom preservation."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "shadows": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Shadows adjustment (-1 to 1)"}),
                "midtones": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Midtones adjustment (-1 to 1)"}),
                "highlights": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Highlights adjustment (-1 to 1)"}),
                "whites": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05,
                    "tooltip": "White point adjustment"}),
                "blacks": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Black point adjustment"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "image/color"
    DESCRIPTION = "Lightroom-style curve controls for shadows, midtones, highlights."
    
    def execute(self, image, shadows: float, midtones: float, 
                highlights: float, whites: float, blacks: float):
        # Early return if all parameters are neutral (identity curve)
        # This preserves full precision by avoiding the 4096-entry LUT
        if shadows == 0.0 and midtones == 0.0 and highlights == 0.0 and whites == 0.0 and blacks == 0.0:
            return (image,)
        
        out = image.clone()
        
        # Build curve from controls (similar to Lightroom's parametric curve)
        # Control points: blacks (0.0), shadows (0.25), midtones (0.5), highlights (0.75), whites (1.0)
        curve_points = [
            (0.0, max(0.0, blacks * 0.1)),
            (0.25, 0.25 + shadows * 0.15),
            (0.5, 0.5 + midtones * 0.2),
            (0.75, 0.75 + highlights * 0.15),
            (1.0, min(1.0, 1.0 + whites * 0.1)),
        ]
        
        # Normalize values above 1.0 temporarily, apply curve, then restore
        max_val = out.max().item()
        if max_val > 1.0:
            # Preserve HDR headroom
            hdr_scale = max_val
            out_normalized = out / hdr_scale
            out_curved = _apply_curves(out_normalized, curve_points)
            out = out_curved * hdr_scale
        else:
            out = _apply_curves(out, curve_points)
        
        return (out,)


# =============================================================================
# NODE: Improved Prepare for High Bit Depth Export
# =============================================================================
class PrepareImageHighBitDepth:
    """Prepares image for high bit-depth export with maximum precision."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "input_is_srgb": ("BOOLEAN", {"default": True,
                    "tooltip": "Input is sRGB (standard ComfyUI). Disable if already linear."}),
                "add_headroom": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.1,
                    "tooltip": "Add highlight headroom (stops above 1.0). Recommended: 0.5-1.0 for grading flexibility."}),
                "deband_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "CRITICAL for quality! Adds noise to break 8-bit quantization. 1.0=cinema-grade, 0=8-bit output."}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "image/processing"
    DESCRIPTION = "REQUIRED for cinema-grade EXR! Converts sRGB→Linear and adds debanding noise to break 8-bit quantization."
    
    def execute(self, image, input_is_srgb: bool, add_headroom: float, deband_strength: float):
        out = image.clone()
        
        # Convert to linear if input is sRGB
        if input_is_srgb:
            out = _srgb_to_linear(out)
        
        # Add highlight headroom (allows values above 1.0 for grading)
        # This scales the image so there's room above diffuse white
        if add_headroom > 0:
            headroom_factor = 1.0 / (1.0 + add_headroom)
            out = out * headroom_factor
        
        # Apply debanding
        if deband_strength > 0:
            out = _deband_image(out, deband_strength)
        
        return (out,)


# =============================================================================
# NODE: Save Image EXR (Improved)
# =============================================================================
class SaveImageEXR:
    """Save image as high-precision EXR with log format options."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "ComfyUI_EXR"}),
                "bit_depth": (["16", "32"], {"default": "32"}),
                "compression": (["none", "zip", "rle", "zips", "piz", "dwaa"], {"default": "zip"}),
                "output_format": (["Linear", "ARRI LogC3", "Sony S-Log3", "Panasonic V-Log", 
                                  "Canon Log 3", "RED Log3G10", "DaVinci Intermediate"],
                                 {"default": "ARRI LogC3",
                                  "tooltip": "Output color space. Log formats recommended for grading."}),
            },
            "optional": {
                "input_is_linear": ("BOOLEAN", {"default": True,
                    "tooltip": "Input is linear (from PrepareImageHighBitDepth). Disable if input is sRGB."}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "image/save"
    OUTPUT_NODE = True
    DESCRIPTION = "Save as 16/32-bit EXR. Default: Log format for professional grading."
    
    def execute(self, image, filename_prefix: str, bit_depth: str, compression: str,
                output_format: str, input_is_linear: bool = True):
        
        # Work with the image
        out = image.clone()
        
        # If input is not linear, convert it first
        if not input_is_linear:
            out = _srgb_to_linear(out)
        
        # Convert to output format
        if output_format == "ARRI LogC3":
            out = _linear_to_log_c(out)
        elif output_format == "Sony S-Log3":
            out = _linear_to_slog3(out)
        elif output_format == "Panasonic V-Log":
            out = _linear_to_vlog(out)
        elif output_format == "Canon Log 3":
            out = _linear_to_clog3(out)
        elif output_format == "RED Log3G10":
            out = _linear_to_redlog3g10(out)
        elif output_format == "DaVinci Intermediate":
            out = _linear_to_davinci_intermediate(out)
        # Linear stays as-is
        
        # Prepare for saving - DO NOT CLAMP to preserve full range
        img = out[0]
        h, w = img.shape[0], img.shape[1]
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), w, h
        )
        filepath = os.path.join(full_output_folder, f"{filename}_{counter:05d}.exr")
        os.makedirs(full_output_folder, exist_ok=True)
        
        # Convert to numpy WITHOUT clamping - preserve all values
        data = img.float().cpu().numpy()
        
        _write_exr(filepath, data, compression, bit_depth, output_format)
        
        return {"ui": {"images": [{"filename": os.path.basename(filepath), "subfolder": subfolder, "type": "output"}]}, 
                "result": (image,)}


# =============================================================================
# NODE: Save Video EXR Sequence (Improved)
# =============================================================================
class SaveVideoEXRSequence:
    """Save video as high-precision EXR sequence with log format options."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "ComfyUI_EXR_seq"}),
                "bit_depth": (["16", "32"], {"default": "32"}),
                "compression": (["none", "zip", "rle", "zips", "piz", "dwaa"], {"default": "zip"}),
                "output_format": (["Linear", "ARRI LogC3", "Sony S-Log3", "Panasonic V-Log",
                                  "Canon Log 3", "RED Log3G10", "DaVinci Intermediate"],
                                 {"default": "ARRI LogC3"}),
            },
            "optional": {
                "input_is_linear": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "execute"
    CATEGORY = "image/video"
    OUTPUT_NODE = True
    DESCRIPTION = "Save video frames as EXR sequence. Default: Log format for grading."
    
    def execute(self, images, filename_prefix: str, bit_depth: str, compression: str,
                output_format: str, input_is_linear: bool = True):
        
        # Process all frames
        out = images.clone()
        
        if not input_is_linear:
            out = _srgb_to_linear(out)
        
        # Apply output format conversion
        if output_format == "ARRI LogC3":
            out = _linear_to_log_c(out)
        elif output_format == "Sony S-Log3":
            out = _linear_to_slog3(out)
        elif output_format == "Panasonic V-Log":
            out = _linear_to_vlog(out)
        elif output_format == "Canon Log 3":
            out = _linear_to_clog3(out)
        elif output_format == "RED Log3G10":
            out = _linear_to_redlog3g10(out)
        elif output_format == "DaVinci Intermediate":
            out = _linear_to_davinci_intermediate(out)
        
        h, w = out.shape[1], out.shape[2]
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), w, h
        )
        os.makedirs(full_output_folder, exist_ok=True)
        
        results = []
        for i in range(out.shape[0]):
            # DO NOT CLAMP - preserve full range
            frame = out[i].float().cpu().numpy()
            path = os.path.join(full_output_folder, f"{filename}_{counter:05d}_{i:06d}.exr")
            _write_exr(path, frame, compression, bit_depth, output_format)
            results.append({"filename": os.path.basename(path), "subfolder": subfolder, "type": "output"})
        
        return {"ui": {"images": results}, "result": (images,)}


# =============================================================================
# NODE: Image Stats (for debugging/verification)
# =============================================================================
class ImageStats:
    """Display image statistics for verification."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("image", "stats",)
    FUNCTION = "execute"
    CATEGORY = "image/analysis"
    DESCRIPTION = "Display image statistics: range, unique values, bit depth estimate."
    
    def execute(self, image):
        img = image[0].float().cpu().numpy()
        
        # Calculate statistics
        min_val = float(img.min())
        max_val = float(img.max())
        mean_val = float(img.mean())
        
        # Count unique values (sample if image is large)
        flat = img.flatten()
        if len(flat) > 100000:
            sample = flat[::len(flat)//100000]
        else:
            sample = flat
        unique_count = len(np.unique(np.round(sample, decimals=6)))
        
        # Estimate above 1.0 percentage
        above_one = float((img > 1.0).mean() * 100)
        
        # Estimate effective bit depth
        if unique_count < 256:
            est_bits = "~8 bits"
        elif unique_count < 4096:
            est_bits = "~12 bits"
        elif unique_count < 16384:
            est_bits = "~14 bits"
        elif unique_count < 65536:
            est_bits = "~16 bits"
        else:
            est_bits = ">16 bits (float)"
        
        stats = f"""Image Statistics:
Range: {min_val:.4f} - {max_val:.4f}
Mean: {mean_val:.4f}
Above 1.0: {above_one:.1f}%
Unique values (sampled): ~{unique_count:,}
Effective bit depth: {est_bits}
Shape: {image.shape}"""
        
        print(stats)
        
        return (image, stats)


# =============================================================================
# NODE: Auto Color Match to Reference
# =============================================================================
class ColorMatchToReference:
    """Automatically match processed image colors/brightness to reference (original) image."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "processed_image": ("IMAGE",),
                "reference_image": ("IMAGE",),
                "match_luminance": ("BOOLEAN", {"default": True,
                    "tooltip": "Match overall brightness levels"}),
                "match_contrast": ("BOOLEAN", {"default": True,
                    "tooltip": "Match contrast/dynamic range"}),
                "match_colors": ("BOOLEAN", {"default": True,
                    "tooltip": "Match color balance (RGB channels)"}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Blend strength (0=no change, 1=full match)"}),
            },
            "optional": {
                "preserve_hdr_headroom": ("BOOLEAN", {"default": True,
                    "tooltip": "Preserve values above 1.0 for HDR"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "image/color"
    DESCRIPTION = "Match processed image colors/brightness to reference image. Prevents dark/shifted outputs."
    
    def execute(self, processed_image, reference_image, 
                match_luminance: bool, match_contrast: bool, match_colors: bool,
                strength: float, preserve_hdr_headroom: bool = True):
        
        if strength == 0:
            return (processed_image,)
        
        proc = processed_image.clone()
        ref = reference_image.clone()
        
        # Work on first frame of each
        proc_frame = proc[0].float()
        ref_frame = ref[0].float()
        
        # Store HDR values (above 1.0) to restore later
        if preserve_hdr_headroom:
            hdr_mask = proc_frame > 1.0
            hdr_values = proc_frame.clone()
        
        # Calculate statistics for reference
        ref_mean = ref_frame.mean(dim=(0, 1))  # Mean per channel
        ref_std = ref_frame.std(dim=(0, 1))    # Std per channel
        ref_luminance = (0.2126 * ref_frame[..., 0] + 0.7152 * ref_frame[..., 1] + 0.0722 * ref_frame[..., 2])
        ref_lum_mean = ref_luminance.mean()
        ref_lum_std = ref_luminance.std()
        
        # Calculate statistics for processed
        proc_mean = proc_frame.mean(dim=(0, 1))
        proc_std = proc_frame.std(dim=(0, 1))
        proc_luminance = (0.2126 * proc_frame[..., 0] + 0.7152 * proc_frame[..., 1] + 0.0722 * proc_frame[..., 2])
        proc_lum_mean = proc_luminance.mean()
        proc_lum_std = proc_luminance.std()
        
        result = proc_frame.clone()
        
        # Match luminance (overall brightness)
        if match_luminance:
            # Shift to match reference mean brightness
            lum_shift = ref_lum_mean - proc_lum_mean
            result = result + lum_shift
        
        # Match contrast (dynamic range)
        if match_contrast and proc_lum_std > 0.001:
            # Scale to match reference contrast
            contrast_scale = ref_lum_std / (proc_lum_std + 1e-6)
            # Apply around the mean
            current_mean = result.mean(dim=(0, 1), keepdim=True)
            result = (result - current_mean) * contrast_scale + current_mean
        
        # Match colors (per-channel)
        if match_colors:
            for c in range(3):
                if proc_std[c] > 0.001:
                    # Normalize then scale to reference distribution
                    channel = result[..., c]
                    channel_mean = channel.mean()
                    channel_std = channel.std()
                    
                    # Standardize
                    normalized = (channel - channel_mean) / (channel_std + 1e-6)
                    # Apply reference statistics
                    matched = normalized * ref_std[c] + ref_mean[c]
                    result[..., c] = matched
        
        # Blend with original based on strength
        if strength < 1.0:
            result = proc_frame * (1 - strength) + result * strength
        
        # Restore HDR headroom
        if preserve_hdr_headroom:
            # Keep values that were above 1.0 in the original processed image
            result = torch.where(hdr_mask, hdr_values, result)
        
        # Ensure no negative values
        result = result.clamp(min=0.0)
        
        # Apply to all frames
        out = proc.clone()
        out[0] = result
        
        # If multiple frames, apply same transform to all
        if proc.shape[0] > 1:
            for i in range(1, proc.shape[0]):
                frame = proc[i].float()
                
                if match_luminance:
                    frame = frame + lum_shift
                
                if match_contrast and proc_lum_std > 0.001:
                    current_mean = frame.mean(dim=(0, 1), keepdim=True)
                    frame = (frame - current_mean) * contrast_scale + current_mean
                
                if match_colors:
                    for c in range(3):
                        if proc_std[c] > 0.001:
                            channel = frame[..., c]
                            channel_mean = channel.mean()
                            channel_std = channel.std()
                            normalized = (channel - channel_mean) / (channel_std + 1e-6)
                            frame[..., c] = normalized * ref_std[c] + ref_mean[c]
                
                if strength < 1.0:
                    frame = proc[i].float() * (1 - strength) + frame * strength
                
                frame = frame.clamp(min=0.0)
                out[i] = frame
        
        return (out,)


# =============================================================================
# NODE: Auto Exposure Match
# =============================================================================
class AutoExposureMatch:
    """Automatically adjust exposure to match reference image brightness."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "reference": ("IMAGE",),
                "method": (["mean", "median", "highlights", "shadows"], {"default": "mean",
                    "tooltip": "What to match: mean brightness, median, highlights, or shadows"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "FLOAT",)
    RETURN_NAMES = ("image", "exposure_adjustment",)
    FUNCTION = "execute"
    CATEGORY = "image/color"
    DESCRIPTION = "Auto-adjust exposure to match reference brightness. Returns adjustment in stops."
    
    def execute(self, image, reference, method: str):
        img = image.clone()
        ref = reference[0].float()
        
        # Calculate luminance
        ref_lum = 0.2126 * ref[..., 0] + 0.7152 * ref[..., 1] + 0.0722 * ref[..., 2]
        
        for i in range(img.shape[0]):
            frame = img[i].float()
            frame_lum = 0.2126 * frame[..., 0] + 0.7152 * frame[..., 1] + 0.0722 * frame[..., 2]
            
            if method == "mean":
                ref_val = ref_lum.mean()
                frame_val = frame_lum.mean()
            elif method == "median":
                ref_val = ref_lum.median()
                frame_val = frame_lum.median()
            elif method == "highlights":
                # Top 10% of values
                ref_val = ref_lum.quantile(0.9)
                frame_val = frame_lum.quantile(0.9)
            else:  # shadows
                # Bottom 10% of values
                ref_val = ref_lum.quantile(0.1)
                frame_val = frame_lum.quantile(0.1)
            
            # Calculate multiplier
            if frame_val > 0.001:
                multiplier = ref_val / frame_val
                img[i] = frame * multiplier
        
        # Calculate exposure adjustment in stops
        if frame_val > 0.001:
            exposure_stops = float(math.log2(multiplier))
        else:
            exposure_stops = 0.0
        
        return (img, exposure_stops)


# =============================================================================
# NODE: Advanced Color Match (5 Algorithms)
# =============================================================================
class AdvancedColorMatch:
    """Advanced automatic color matching with 5 professional algorithms."""
    
    METHODS = [
        "Histogram Matching",
        "LAB Color Space",
        "Reinhard Transfer",
        "CLAHE + Histogram",
        "CDF Matching"
    ]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "processed_image": ("IMAGE",),
                "reference_image": ("IMAGE",),
                "method": (cls.METHODS, {"default": "Histogram Matching",
                    "tooltip": "Algorithm for color matching"}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Blend strength (0=no change, 1=full match)"}),
            },
            "optional": {
                "match_luminance_only": ("BOOLEAN", {"default": False,
                    "tooltip": "Only match brightness, preserve original colors"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("image", "method_info",)
    FUNCTION = "execute"
    CATEGORY = "image/color"
    DESCRIPTION = "Advanced color matching: Histogram, LAB, Reinhard, CLAHE, or CDF algorithms."
    
    def _histogram_match_channel(self, source: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Match histogram of source to reference for a single channel."""
        # Get histograms
        src_values, src_counts = np.unique(source.flatten(), return_counts=True)
        ref_values, ref_counts = np.unique(reference.flatten(), return_counts=True)
        
        # Calculate CDFs
        src_cdf = np.cumsum(src_counts).astype(np.float64)
        src_cdf /= src_cdf[-1]
        
        ref_cdf = np.cumsum(ref_counts).astype(np.float64)
        ref_cdf /= ref_cdf[-1]
        
        # Create mapping
        interp_values = np.interp(src_cdf, ref_cdf, ref_values)
        
        # Map source values to matched values
        result = np.interp(source.flatten(), src_values, interp_values)
        return result.reshape(source.shape)
    
    def _histogram_matching(self, proc: np.ndarray, ref: np.ndarray) -> np.ndarray:
        """Per-channel histogram matching."""
        result = np.zeros_like(proc)
        for c in range(3):
            result[..., c] = self._histogram_match_channel(proc[..., c], ref[..., c])
        return result
    
    def _rgb_to_lab(self, rgb: np.ndarray) -> np.ndarray:
        """Convert RGB to LAB color space."""
        if _HAS_CV2:
            # OpenCV expects BGR and 0-255 range for proper conversion
            bgr = cv2.cvtColor((rgb * 255).astype(np.float32), cv2.COLOR_RGB2BGR)
            lab = cv2.cvtColor(bgr.astype(np.uint8), cv2.COLOR_BGR2LAB)
            return lab.astype(np.float32)
        else:
            # Fallback: simple approximation
            l = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
            a = rgb[..., 0] - rgb[..., 1]
            b = rgb[..., 2] - rgb[..., 1]
            return np.stack([l * 100, a * 128 + 128, b * 128 + 128], axis=-1)
    
    def _lab_to_rgb(self, lab: np.ndarray) -> np.ndarray:
        """Convert LAB to RGB color space."""
        if _HAS_CV2:
            bgr = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            return rgb.astype(np.float32) / 255.0
        else:
            # Fallback
            l = lab[..., 0] / 100
            return np.stack([l, l, l], axis=-1)
    
    def _lab_matching(self, proc: np.ndarray, ref: np.ndarray) -> np.ndarray:
        """Match in LAB color space for better perceptual results."""
        proc_lab = self._rgb_to_lab(proc)
        ref_lab = self._rgb_to_lab(ref)
        
        # Match each LAB channel
        result_lab = np.zeros_like(proc_lab)
        for c in range(3):
            result_lab[..., c] = self._histogram_match_channel(proc_lab[..., c], ref_lab[..., c])
        
        return self._lab_to_rgb(result_lab)
    
    def _reinhard_transfer(self, proc: np.ndarray, ref: np.ndarray) -> np.ndarray:
        """Reinhard color transfer algorithm (mean/std in LAB space)."""
        proc_lab = self._rgb_to_lab(proc)
        ref_lab = self._rgb_to_lab(ref)
        
        result_lab = np.zeros_like(proc_lab)
        
        for c in range(3):
            proc_mean = proc_lab[..., c].mean()
            proc_std = proc_lab[..., c].std() + 1e-6
            ref_mean = ref_lab[..., c].mean()
            ref_std = ref_lab[..., c].std() + 1e-6
            
            # Normalize then apply reference statistics
            normalized = (proc_lab[..., c] - proc_mean) / proc_std
            result_lab[..., c] = normalized * ref_std + ref_mean
        
        return self._lab_to_rgb(np.clip(result_lab, 0, 255))
    
    def _clahe_histogram(self, proc: np.ndarray, ref: np.ndarray) -> np.ndarray:
        """CLAHE for local contrast + histogram matching for global."""
        if not _HAS_CV2:
            return self._histogram_matching(proc, ref)
        
        # Convert to LAB
        proc_uint8 = (np.clip(proc, 0, 1) * 255).astype(np.uint8)
        proc_bgr = cv2.cvtColor(proc_uint8, cv2.COLOR_RGB2BGR)
        proc_lab = cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2LAB)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        proc_lab[..., 0] = clahe.apply(proc_lab[..., 0])
        
        # Convert back
        proc_bgr = cv2.cvtColor(proc_lab, cv2.COLOR_LAB2BGR)
        proc_rgb = cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        # Then apply histogram matching
        return self._histogram_matching(proc_rgb, ref)
    
    def _cdf_matching(self, proc: np.ndarray, ref: np.ndarray) -> np.ndarray:
        """Precise CDF (Cumulative Distribution Function) matching."""
        result = np.zeros_like(proc)
        
        for c in range(3):
            # Flatten channels
            src = proc[..., c].flatten()
            tgt = ref[..., c].flatten()
            
            # Sort and get indices
            src_sorted_idx = np.argsort(src)
            tgt_sorted = np.sort(tgt)
            
            # Create rank-based mapping
            ranks = np.empty_like(src_sorted_idx)
            ranks[src_sorted_idx] = np.linspace(0, len(tgt_sorted) - 1, len(src)).astype(int)
            
            # Map to target values
            matched = tgt_sorted[ranks]
            result[..., c] = matched.reshape(proc[..., c].shape)
        
        return result
    
    def execute(self, processed_image, reference_image, method: str, strength: float,
                match_luminance_only: bool = False):
        
        if strength == 0:
            return (processed_image, f"Method: {method} (strength=0, no change)")
        
        proc = processed_image.clone()
        ref = reference_image[0].float().cpu().numpy()
        
        # Clamp reference to 0-1 for algorithm stability
        ref = np.clip(ref, 0, 1)
        
        results = []
        
        for i in range(proc.shape[0]):
            frame = proc[i].float().cpu().numpy()
            frame_clipped = np.clip(frame, 0, 1)
            
            # Apply selected method
            if method == "Histogram Matching":
                matched = self._histogram_matching(frame_clipped, ref)
            elif method == "LAB Color Space":
                matched = self._lab_matching(frame_clipped, ref)
            elif method == "Reinhard Transfer":
                matched = self._reinhard_transfer(frame_clipped, ref)
            elif method == "CLAHE + Histogram":
                matched = self._clahe_histogram(frame_clipped, ref)
            elif method == "CDF Matching":
                matched = self._cdf_matching(frame_clipped, ref)
            else:
                matched = frame_clipped
            
            # If luminance only, preserve original colors
            if match_luminance_only:
                orig_lum = 0.2126 * frame_clipped[..., 0] + 0.7152 * frame_clipped[..., 1] + 0.0722 * frame_clipped[..., 2]
                new_lum = 0.2126 * matched[..., 0] + 0.7152 * matched[..., 1] + 0.0722 * matched[..., 2]
                
                # Scale original colors by luminance ratio
                lum_ratio = (new_lum + 1e-6) / (orig_lum + 1e-6)
                matched = frame_clipped * lum_ratio[..., np.newaxis]
            
            # Blend with original
            if strength < 1.0:
                matched = frame_clipped * (1 - strength) + matched * strength
            
            # Restore HDR values above 1.0 from original
            hdr_mask = frame > 1.0
            matched = np.where(hdr_mask, frame, matched)
            
            results.append(torch.from_numpy(matched.astype(np.float32)))
        
        out = torch.stack(results)
        
        info = f"Method: {method}, Strength: {strength:.0%}"
        if match_luminance_only:
            info += " (luminance only)"
        
        return (out, info)


# =============================================================================
# IMPORT ADDITIONAL NODE MODULES
# =============================================================================
from .cinema_delivery import (
    CINEMA_DELIVERY_NODES,
    CINEMA_DELIVERY_DISPLAY_NAMES,
)
from .animated_motion import (
    ANIMATED_MOTION_NODES,
    ANIMATED_MOTION_DISPLAY_NAMES,
)
from .luminance_deflicker import (
    LUMINANCE_DEFLICKER_NODES,
    LUMINANCE_DEFLICKER_DISPLAY_NAMES,
)
from .exposure_bracketing import (
    EXPOSURE_BRACKETING_NODES,
    EXPOSURE_BRACKETING_DISPLAY_NAMES,
)
from .shadow_controlled_hdr import (
    SHADOW_HDR_NODES,
    SHADOW_HDR_DISPLAY_NAMES,
)
from .fractal_bitdepth import (
    FRACTAL_BITDEPTH_NODES,
    FRACTAL_BITDEPTH_DISPLAY_NAMES,
)
from .scene_segmentation import (
    SCENE_SEGMENTATION_NODES,
    SCENE_SEGMENTATION_DISPLAY_NAMES,
)
from .layer_processor import (
    LAYER_PROCESSOR_NODES,
    LAYER_PROCESSOR_DISPLAY_NAMES,
)
from .ai_detail_layer import (
    AI_DETAIL_NODES,
    AI_DETAIL_DISPLAY_NAMES,
)
from .layer_assembly import (
    LAYER_ASSEMBLY_NODES,
    LAYER_ASSEMBLY_DISPLAY_NAMES,
)
from .video_encoder import (
    VIDEO_ENCODER_NODES,
    VIDEO_ENCODER_DISPLAY_NAMES,
)
from .temporal_fractal import (
    TEMPORAL_FRACTAL_NODES,
    TEMPORAL_FRACTAL_DISPLAY_NAMES,
)
from .frame_interpolator import (
    FRAME_INTERP_NODES,
    FRAME_INTERP_DISPLAY_NAMES,
)

# =============================================================================
# NODE MAPPINGS
# =============================================================================
NODE_CLASS_MAPPINGS = {
    # Core Color/EXR Nodes
    "ColorSpaceConverter": ColorSpaceConverter,
    "ColorGradingController": ColorGradingController,
    "HDRCurveEditor": HDRCurveEditor,
    "PrepareImageHighBitDepth": PrepareImageHighBitDepth,
    "SaveImageEXR": SaveImageEXR,
    "SaveVideoEXRSequence": SaveVideoEXRSequence,
    "ImageStats": ImageStats,
    "ColorMatchToReference": ColorMatchToReference,
    "AutoExposureMatch": AutoExposureMatch,
    "AdvancedColorMatch": AdvancedColorMatch,
}

# Add Cinema Delivery nodes
NODE_CLASS_MAPPINGS.update(CINEMA_DELIVERY_NODES)

# Add Animated Motion nodes
NODE_CLASS_MAPPINGS.update(ANIMATED_MOTION_NODES)

# Add Luminance Deflicker nodes
NODE_CLASS_MAPPINGS.update(LUMINANCE_DEFLICKER_NODES)

# Add Exposure Bracketing nodes
NODE_CLASS_MAPPINGS.update(EXPOSURE_BRACKETING_NODES)

# Add Shadow-Controlled HDR nodes
NODE_CLASS_MAPPINGS.update(SHADOW_HDR_NODES)

# Add Fractal Bit-Depth Expansion nodes
NODE_CLASS_MAPPINGS.update(FRACTAL_BITDEPTH_NODES)

# Add Scene Segmentation nodes
NODE_CLASS_MAPPINGS.update(SCENE_SEGMENTATION_NODES)

# Add Layer Processor nodes
NODE_CLASS_MAPPINGS.update(LAYER_PROCESSOR_NODES)

# Add AI Detail Enhancement nodes
NODE_CLASS_MAPPINGS.update(AI_DETAIL_NODES)

# Add Layer Assembly nodes
NODE_CLASS_MAPPINGS.update(LAYER_ASSEMBLY_NODES)

# Add Video Encoder nodes
NODE_CLASS_MAPPINGS.update(VIDEO_ENCODER_NODES)

# Add Temporal Fractal nodes
NODE_CLASS_MAPPINGS.update(TEMPORAL_FRACTAL_NODES)

# Add Frame Interpolator nodes
NODE_CLASS_MAPPINGS.update(FRAME_INTERP_NODES)

NODE_DISPLAY_NAME_MAPPINGS = {
    "ColorSpaceConverter": "Color Space Converter",
    "ColorGradingController": "Color Grading Controller",
    "HDRCurveEditor": "HDR Curve Editor",
    "PrepareImageHighBitDepth": "Prepare Image High Bit Depth",
    "SaveImageEXR": "Save Image EXR",
    "SaveVideoEXRSequence": "Save Video EXR Sequence",
    "ImageStats": "Image Stats",
    "ColorMatchToReference": "Color Match to Reference",
    "AutoExposureMatch": "Auto Exposure Match",
    "AdvancedColorMatch": "Advanced Color Match",
}

# Add Cinema Delivery display names
NODE_DISPLAY_NAME_MAPPINGS.update(CINEMA_DELIVERY_DISPLAY_NAMES)

# Add Animated Motion display names
NODE_DISPLAY_NAME_MAPPINGS.update(ANIMATED_MOTION_DISPLAY_NAMES)

# Add Luminance Deflicker display names
NODE_DISPLAY_NAME_MAPPINGS.update(LUMINANCE_DEFLICKER_DISPLAY_NAMES)

# Add Exposure Bracketing display names
NODE_DISPLAY_NAME_MAPPINGS.update(EXPOSURE_BRACKETING_DISPLAY_NAMES)

# Add Shadow-Controlled HDR display names
NODE_DISPLAY_NAME_MAPPINGS.update(SHADOW_HDR_DISPLAY_NAMES)

# Add Fractal Bit-Depth display names
NODE_DISPLAY_NAME_MAPPINGS.update(FRACTAL_BITDEPTH_DISPLAY_NAMES)

# Add Scene Segmentation display names
NODE_DISPLAY_NAME_MAPPINGS.update(SCENE_SEGMENTATION_DISPLAY_NAMES)

# Add Layer Processor display names
NODE_DISPLAY_NAME_MAPPINGS.update(LAYER_PROCESSOR_DISPLAY_NAMES)

# Add AI Detail Enhancement display names
NODE_DISPLAY_NAME_MAPPINGS.update(AI_DETAIL_DISPLAY_NAMES)

# Add Layer Assembly display names
NODE_DISPLAY_NAME_MAPPINGS.update(LAYER_ASSEMBLY_DISPLAY_NAMES)

# Add Video Encoder display names
NODE_DISPLAY_NAME_MAPPINGS.update(VIDEO_ENCODER_DISPLAY_NAMES)

# Add Temporal Fractal display names
NODE_DISPLAY_NAME_MAPPINGS.update(TEMPORAL_FRACTAL_DISPLAY_NAMES)

# Add Frame Interpolator display names
NODE_DISPLAY_NAME_MAPPINGS.update(FRAME_INTERP_DISPLAY_NAMES)

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
