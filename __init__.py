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
                "working_space": (["Linear", "sRGB"], {"default": "Linear",
                    "tooltip": "Color space for grading operations"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "image/color"
    DESCRIPTION = "Professional color grading: exposure, contrast, lift/gamma/gain."
    
    def execute(self, image, exposure: float, contrast: float, 
                lift: float, gamma: float, gain: float, saturation: float,
                working_space: str = "Linear"):
        out = image.clone()
        
        # Convert to linear for proper grading if needed
        if working_space == "Linear":
            # Assume input is sRGB from ComfyUI
            out = _srgb_to_linear(out)
        
        # Apply grading operations
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
                "deband_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Debanding strength (0=off). Use 0.3-0.5 for subtle, 1.0+ for aggressive."}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "image/processing"
    DESCRIPTION = "Prepare image for EXR export with proper linear conversion and headroom."
    
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
# NODE MAPPINGS
# =============================================================================
NODE_CLASS_MAPPINGS = {
    "ColorSpaceConverter": ColorSpaceConverter,
    "ColorGradingController": ColorGradingController,
    "HDRCurveEditor": HDRCurveEditor,
    "PrepareImageHighBitDepth": PrepareImageHighBitDepth,
    "SaveImageEXR": SaveImageEXR,
    "SaveVideoEXRSequence": SaveVideoEXRSequence,
    "ImageStats": ImageStats,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ColorSpaceConverter": "Color Space Converter",
    "ColorGradingController": "Color Grading Controller",
    "HDRCurveEditor": "HDR Curve Editor",
    "PrepareImageHighBitDepth": "Prepare Image High Bit Depth",
    "SaveImageEXR": "Save Image EXR",
    "SaveVideoEXRSequence": "Save Video EXR Sequence",
    "ImageStats": "Image Stats",
}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
