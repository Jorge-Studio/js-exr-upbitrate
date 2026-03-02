# Shadow-Controlled HDR Processing for ComfyUI
# Uses colour-science library for accurate color transforms
# Power curve shadow control to eliminate noise artifacts
# Part of js-exr-upbitrate package

import os
import numpy as np

try:
    import torch
except ImportError:
    torch = None

try:
    import colour
    from colour.models import RGB_COLOURSPACES
    _HAS_COLOUR = True
except ImportError:
    _HAS_COLOUR = False
    print("[WARNING] colour-science not installed. Install with: pip install colour-science")


class ShadowControlledExposure:
    """
    Generate exposure brackets with proper color science and power curve shadow control.
    
    Key improvements:
    - Uses colour-science for accurate Rec.709 handling
    - Power curves (not log) for shadow control to prevent noise amplification
    - Values "peak" at the curve limits instead of rolling off
    - Proper EOTF/OETF handling for Rec.709
    """
    
    CATEGORY = "image/hdr"
    FUNCTION = "generate_brackets"
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("ev_plus_4", "ev_plus_2", "ev_0", "ev_minus_2", "ev_minus_4")
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "exposure_step": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.5,
                    "max": 4.0,
                    "step": 0.5,
                    "tooltip": "Exposure step between brackets in stops"
                }),
                "shadow_power": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.5,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Power curve exponent for shadows - higher = more shadow suppression"
                }),
                "shadow_threshold": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.01,
                    "max": 0.5,
                    "step": 0.01,
                    "tooltip": "Below this value, apply shadow power curve"
                }),
                "peak_clipping": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Hard clip values at peak instead of soft rolloff"
                }),
                "denoise_shadows": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Apply shadow denoising before processing"
                }),
            },
            "optional": {
                "input_colorspace": (["sRGB", "Rec.709", "Linear", "ACEScg", "ACES2065-1"], {
                    "default": "sRGB",
                    "tooltip": "Input color space"
                }),
                "working_colorspace": (["Rec.709", "sRGB", "Linear"], {
                    "default": "Rec.709",
                    "tooltip": "Working color space for processing"
                }),
                "shadow_denoise_strength": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                }),
            }
        }
    
    def _to_linear(self, img, colorspace):
        """Convert to linear using colour-science."""
        if not _HAS_COLOUR:
            # Fallback: simple gamma
            if colorspace in ["sRGB", "Rec.709"]:
                return np.where(img <= 0.04045, img / 12.92, 
                               np.power((img + 0.055) / 1.055, 2.4))
            return img
        
        img_clipped = np.clip(img, 0, 1)
        
        if colorspace == "sRGB":
            return colour.cctf_decoding(img_clipped, function='sRGB')
        elif colorspace == "Rec.709":
            return colour.cctf_decoding(img_clipped, function='ITU-R BT.709')
        elif colorspace == "ACEScg":
            # ACEScg is already linear, just need to convert primaries
            return img_clipped
        elif colorspace == "ACES2065-1":
            return img_clipped
        else:
            return img_clipped
    
    def _from_linear(self, img, colorspace):
        """Convert from linear using colour-science."""
        if not _HAS_COLOUR:
            # Fallback: simple gamma
            if colorspace in ["sRGB", "Rec.709"]:
                return np.where(img <= 0.0031308, img * 12.92,
                               1.055 * np.power(img, 1/2.4) - 0.055)
            return img
        
        img_clipped = np.clip(img, 0, 1)
        
        if colorspace == "sRGB":
            return colour.cctf_encoding(img_clipped, function='sRGB')
        elif colorspace == "Rec.709":
            return colour.cctf_encoding(img_clipped, function='ITU-R BT.709')
        else:
            return img_clipped
    
    def _apply_shadow_power_curve(self, img, power, threshold):
        """
        Apply power curve to shadows to suppress noise.
        
        Power curves are better than log because:
        - They have a defined peak (no rolloff)
        - They compress shadows non-linearly
        - Values above threshold pass through unchanged
        """
        # Create shadow mask
        luminance = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
        shadow_mask = luminance < threshold
        
        # Apply power curve only to shadows
        result = img.copy()
        
        for c in range(3):
            channel = img[..., c]
            
            # Power curve: (value / threshold)^power * threshold
            # This compresses shadows while maintaining threshold as the peak
            shadow_values = channel[shadow_mask]
            if shadow_values.size > 0:
                # Normalize to threshold, apply power, denormalize
                normalized = shadow_values / threshold
                powered = np.power(np.clip(normalized, 0, 1), power)
                result[..., c][shadow_mask] = powered * threshold
        
        return result
    
    def _denoise_shadows(self, img, strength, threshold):
        """
        Denoise shadow regions using local averaging.
        Only affects dark regions to preserve detail in midtones/highlights.
        """
        try:
            from scipy.ndimage import gaussian_filter
        except ImportError:
            return img
        
        luminance = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
        
        # Shadow blend mask - smooth transition
        shadow_blend = np.clip((threshold - luminance) / threshold, 0, 1)
        shadow_blend = np.power(shadow_blend, 0.5)  # Soften the mask
        
        result = img.copy()
        
        for c in range(3):
            # Blur only the shadow regions
            blurred = gaussian_filter(img[..., c], sigma=1.5)
            # Blend based on shadow mask and strength
            result[..., c] = img[..., c] * (1 - shadow_blend * strength) + blurred * (shadow_blend * strength)
        
        return result
    
    def _apply_exposure(self, img_linear, stops, peak_clip):
        """Apply exposure in linear space with peak clipping."""
        multiplier = 2.0 ** stops
        
        exposed = img_linear * multiplier
        
        if peak_clip:
            # Hard clip at 1.0 - values PEAK at the limit
            exposed = np.clip(exposed, 0, 1)
        else:
            # Soft shoulder - values roll off (can cause artifacts)
            # Use simple Reinhard-style compression
            exposed = exposed / (exposed + 1) * 2
            exposed = np.clip(exposed, 0, 1)
        
        return exposed.astype(np.float32)
    
    def generate_brackets(self, image, exposure_step, shadow_power, shadow_threshold,
                          peak_clipping, denoise_shadows,
                          input_colorspace="sRGB", working_colorspace="Rec.709",
                          shadow_denoise_strength=0.5):
        
        if hasattr(image, 'cpu'):
            img_np = image.cpu().float().numpy()
        else:
            img_np = np.array(image, dtype=np.float32)
        
        # Handle batch dimension
        if len(img_np.shape) == 3:
            img_np = img_np[np.newaxis, ...]
        
        all_brackets = [[], [], [], [], []]
        stops = [exposure_step * 2, exposure_step, 0, -exposure_step, -exposure_step * 2]
        
        for frame_idx in range(img_np.shape[0]):
            frame = img_np[frame_idx]
            
            # 1. Convert to linear Rec.709
            frame_linear = self._to_linear(frame, input_colorspace)
            
            # 2. Apply shadow denoising BEFORE exposure changes
            if denoise_shadows:
                frame_linear = self._denoise_shadows(frame_linear, shadow_denoise_strength, shadow_threshold)
            
            # 3. Apply shadow power curve to suppress noise in darks
            frame_shadow_controlled = self._apply_shadow_power_curve(frame_linear, shadow_power, shadow_threshold)
            
            # 4. Generate brackets
            for i, ev in enumerate(stops):
                # Apply exposure in linear space
                exposed = self._apply_exposure(frame_shadow_controlled, ev, peak_clipping)
                
                # For underexposed brackets, apply additional shadow control
                if ev < 0:
                    # The underexposed image will have MORE shadows, apply stronger control
                    exposed = self._apply_shadow_power_curve(exposed, shadow_power * (1 + abs(ev)/4), shadow_threshold)
                
                # Convert back to working colorspace
                bracket = self._from_linear(exposed, working_colorspace)
                all_brackets[i].append(bracket)
        
        # Stack frames
        results = []
        for bracket_list in all_brackets:
            stacked = np.stack(bracket_list, axis=0).astype(np.float32)
            if torch:
                results.append(torch.from_numpy(stacked))
            else:
                results.append(stacked)
        
        print(f"[ShadowControlledExposure] Generated {len(results)} brackets with shadow power={shadow_power}")
        
        return tuple(results)


class ShadowCurveProcessor:
    """
    Apply power curve shadow control to existing images.
    Useful for post-processing HDR merged results to clean up shadow noise.
    """
    
    CATEGORY = "image/hdr"
    FUNCTION = "process"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("processed",)
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "shadow_power": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.5,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Power curve exponent - higher = more shadow compression"
                }),
                "shadow_threshold": ("FLOAT", {
                    "default": 0.15,
                    "min": 0.01,
                    "max": 0.5,
                    "step": 0.01,
                    "tooltip": "Luminance threshold defining shadow region"
                }),
                "blend_width": ("FLOAT", {
                    "default": 0.05,
                    "min": 0.01,
                    "max": 0.2,
                    "step": 0.01,
                    "tooltip": "Width of transition zone between shadow and non-shadow"
                }),
                "preserve_color": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Preserve color ratios in shadows"
                }),
            },
            "optional": {
                "input_colorspace": (["sRGB", "Rec.709", "Linear"], {
                    "default": "sRGB"
                }),
            }
        }
    
    def _to_linear(self, img, colorspace):
        """Convert to linear."""
        if not _HAS_COLOUR:
            if colorspace in ["sRGB", "Rec.709"]:
                return np.where(img <= 0.04045, img / 12.92, 
                               np.power((img + 0.055) / 1.055, 2.4))
            return img
        
        img_clipped = np.clip(img, 0, 1)
        
        if colorspace == "sRGB":
            return colour.cctf_decoding(img_clipped, function='sRGB')
        elif colorspace == "Rec.709":
            return colour.cctf_decoding(img_clipped, function='ITU-R BT.709')
        return img_clipped
    
    def _from_linear(self, img, colorspace):
        """Convert from linear."""
        if not _HAS_COLOUR:
            if colorspace in ["sRGB", "Rec.709"]:
                return np.where(img <= 0.0031308, img * 12.92,
                               1.055 * np.power(np.clip(img, 0.0001, None), 1/2.4) - 0.055)
            return img
        
        img_clipped = np.clip(img, 0, 1)
        
        if colorspace == "sRGB":
            return colour.cctf_encoding(img_clipped, function='sRGB')
        elif colorspace == "Rec.709":
            return colour.cctf_encoding(img_clipped, function='ITU-R BT.709')
        return img_clipped
    
    def process(self, image, shadow_power, shadow_threshold, blend_width, 
                preserve_color, input_colorspace="sRGB"):
        
        if hasattr(image, 'cpu'):
            img_np = image.cpu().float().numpy()
        else:
            img_np = np.array(image, dtype=np.float32)
        
        if len(img_np.shape) == 3:
            img_np = img_np[np.newaxis, ...]
        
        results = []
        
        for frame_idx in range(img_np.shape[0]):
            frame = img_np[frame_idx]
            
            # Convert to linear
            frame_linear = self._to_linear(frame, input_colorspace)
            
            # Calculate luminance
            luminance = 0.2126 * frame_linear[..., 0] + 0.7152 * frame_linear[..., 1] + 0.0722 * frame_linear[..., 2]
            
            # Create smooth blend mask
            blend_start = shadow_threshold - blend_width
            blend_end = shadow_threshold + blend_width
            
            # Smooth transition mask
            blend_mask = np.clip((luminance - blend_start) / (blend_end - blend_start), 0, 1)
            blend_mask = 1 - blend_mask  # Invert so shadows = 1
            
            result = frame_linear.copy()
            
            if preserve_color:
                # Apply power curve to luminance only, preserve color ratios
                # Avoid division by zero
                safe_lum = np.maximum(luminance, 1e-6)
                
                # Power curve on luminance
                new_lum = np.where(
                    luminance < shadow_threshold,
                    np.power(luminance / shadow_threshold, shadow_power) * shadow_threshold,
                    luminance
                )
                
                # Scale RGB by luminance ratio
                lum_ratio = new_lum / safe_lum
                
                for c in range(3):
                    # Blend between original and power-curved based on mask
                    curved = frame_linear[..., c] * lum_ratio
                    result[..., c] = frame_linear[..., c] * (1 - blend_mask) + curved * blend_mask
            else:
                # Apply power curve directly to each channel
                for c in range(3):
                    channel = frame_linear[..., c]
                    
                    # Power curve
                    curved = np.where(
                        channel < shadow_threshold,
                        np.power(channel / shadow_threshold, shadow_power) * shadow_threshold,
                        channel
                    )
                    
                    # Blend
                    result[..., c] = channel * (1 - blend_mask) + curved * blend_mask
            
            # Clip and convert back
            result = np.clip(result, 0, 1)
            result = self._from_linear(result, input_colorspace)
            
            results.append(result)
        
        output = np.stack(results, axis=0).astype(np.float32)
        
        if torch:
            output = torch.from_numpy(output)
        
        print(f"[ShadowCurveProcessor] Processed {len(results)} frames with power={shadow_power}")
        
        return (output,)


class Rec709Converter:
    """
    Convert images to/from Rec.709 using colour-science.
    Essential for preventing color space artifacts before HDR processing.
    """
    
    CATEGORY = "image/color"
    FUNCTION = "convert"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("converted",)
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "source_colorspace": (["sRGB", "Rec.709", "Linear sRGB", "ACEScg", "ACES2065-1", "DCI-P3", "Display P3"], {
                    "default": "sRGB"
                }),
                "target_colorspace": (["Rec.709", "sRGB", "Linear sRGB", "Linear Rec.709", "ACEScg"], {
                    "default": "Rec.709"
                }),
                "chromatic_adaptation": (["Bradford", "Von Kries", "None"], {
                    "default": "Bradford",
                    "tooltip": "Chromatic adaptation transform for white point conversion"
                }),
            }
        }
    
    def convert(self, image, source_colorspace, target_colorspace, chromatic_adaptation):
        if not _HAS_COLOUR:
            print("[WARNING] colour-science not available, returning unchanged image")
            return (image,)
        
        if hasattr(image, 'cpu'):
            img_np = image.cpu().float().numpy()
        else:
            img_np = np.array(image, dtype=np.float32)
        
        if len(img_np.shape) == 3:
            img_np = img_np[np.newaxis, ...]
        
        results = []
        
        # Get colorspace objects
        source_cs = self._get_colorspace(source_colorspace)
        target_cs = self._get_colorspace(target_colorspace)
        
        cat = chromatic_adaptation if chromatic_adaptation != "None" else None
        
        for frame_idx in range(img_np.shape[0]):
            frame = np.clip(img_np[frame_idx], 0, 1)
            
            # Convert to XYZ
            if source_colorspace.startswith("Linear"):
                linear = frame
            else:
                # Decode OETF/EOTF
                linear = self._decode(frame, source_colorspace)
            
            # Use the new colour-science API (0.4.3+)
            # colour.RGB_to_XYZ now uses colourspace parameter
            try:
                XYZ = colour.RGB_to_XYZ(
                    linear, 
                    colourspace=source_cs,
                )
                target_RGB = colour.XYZ_to_RGB(
                    XYZ,
                    colourspace=target_cs,
                )
            except TypeError:
                # Fallback for older API
                XYZ = colour.RGB_to_XYZ(
                    linear, 
                    source_cs.whitepoint,
                    source_cs.whitepoint,
                    source_cs.matrix_RGB_to_XYZ
                )
                target_RGB = colour.XYZ_to_RGB(
                    XYZ,
                    target_cs.whitepoint,
                    target_cs.whitepoint,
                    target_cs.matrix_XYZ_to_RGB
                )
            
            # Encode with target OETF/EOTF if needed
            if target_colorspace.startswith("Linear"):
                output = target_RGB
            else:
                output = self._encode(target_RGB, target_colorspace)
            
            output = np.clip(output, 0, 1).astype(np.float32)
            results.append(output)
        
        output = np.stack(results, axis=0)
        
        if torch:
            output = torch.from_numpy(output)
        
        print(f"[Rec709Converter] Converted {len(results)} frames: {source_colorspace} → {target_colorspace}")
        
        return (output,)
    
    def _get_colorspace(self, name):
        """Get colour-science colorspace object."""
        mapping = {
            "sRGB": "sRGB",
            "Rec.709": "ITU-R BT.709",
            "Linear sRGB": "sRGB",
            "Linear Rec.709": "ITU-R BT.709",
            "ACEScg": "ACEScg",
            "ACES2065-1": "ACES2065-1",
            "DCI-P3": "DCI-P3",
            "Display P3": "Display P3",
        }
        cs_name = mapping.get(name, "sRGB")
        return RGB_COLOURSPACES[cs_name]
    
    def _decode(self, img, colorspace):
        """Decode gamma/OETF to linear."""
        img = np.clip(img, 0, 1)
        if colorspace in ["sRGB", "Linear sRGB"]:
            return colour.cctf_decoding(img, function='sRGB')
        elif colorspace in ["Rec.709", "Linear Rec.709"]:
            return colour.cctf_decoding(img, function='ITU-R BT.709')
        elif colorspace in ["Display P3"]:
            return colour.cctf_decoding(img, function='sRGB')  # Display P3 uses sRGB gamma
        return img
    
    def _encode(self, img, colorspace):
        """Encode linear to gamma/OETF."""
        img = np.clip(img, 0, 1)
        if colorspace in ["sRGB"]:
            return colour.cctf_encoding(img, function='sRGB')
        elif colorspace in ["Rec.709"]:
            return colour.cctf_encoding(img, function='ITU-R BT.709')
        return img


# Node exports
SHADOW_HDR_NODES = {
    "ShadowControlledExposure": ShadowControlledExposure,
    "ShadowCurveProcessor": ShadowCurveProcessor,
    "Rec709Converter": Rec709Converter,
}

SHADOW_HDR_DISPLAY_NAMES = {
    "ShadowControlledExposure": "Shadow-Controlled Exposure Brackets",
    "ShadowCurveProcessor": "Shadow Power Curve",
    "Rec709Converter": "Rec.709 Color Space Converter",
}
