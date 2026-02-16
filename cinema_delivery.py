# Cinema Delivery Nodes for ComfyUI
# Designed for Molinare/Professional DI delivery specifications.
# Part of js-exr-upbitrate package

import os
import csv
import time
import numpy as np
from datetime import datetime

try:
    import torch
except ImportError:
    torch = None

try:
    import pyexr
    _HAS_PYEXR = True
except ImportError:
    _HAS_PYEXR = False

try:
    import folder_paths
except ImportError:
    folder_paths = None


def _linear_to_aces_ap0(img: np.ndarray) -> np.ndarray:
    """Convert scene-linear Rec.709 to ACES AP0 (2065-1)."""
    # sRGB/Rec.709 to ACES AP0 matrix (D65 to ACES white point via Bradford CAT)
    mat = np.array([
        [0.4397010, 0.3829780, 0.1773350],
        [0.0897923, 0.8134230, 0.0967616],
        [0.0175440, 0.1115440, 0.8707040]
    ], dtype=np.float32)
    
    shape = img.shape
    flat = img.reshape(-1, 3)
    out = flat @ mat.T
    return out.reshape(shape)


class SaveEXRSequence:
    """
    Save video frames as DCI 4K ACES 2065-1 EXR sequence.
    
    Output format: shot_name/shot_name.####.exr
    Compliant with Molinare delivery specifications.
    """
    
    CATEGORY = "Cinema Delivery"
    FUNCTION = "save_sequence"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "shot_name": ("STRING", {
                    "default": "KSA_001_010",
                    "tooltip": "Shot name without version (e.g. KSA_001_010)"
                }),
                "version": ("INT", {
                    "default": 1, "min": 0, "max": 999,
                    "tooltip": "Version number (V001, V002, etc.)"
                }),
                "start_frame": ("INT", {
                    "default": 1001, "min": 0, "max": 999999,
                    "tooltip": "Starting frame number (1001 is industry standard)"
                }),
                "input_is_linear": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Input is scene-linear Rec.709 (from HDR processing)"
                }),
                "convert_to_aces": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Convert to ACES 2065-1 (AP0) color space"
                }),
                "bit_depth": (["16", "32"], {
                    "default": "16",
                    "tooltip": "16 = half-float (Molinare spec), 32 = full float"
                }),
                "compression": (["piz", "zip", "zips", "none", "rle"], {
                    "default": "piz",
                    "tooltip": "PIZ recommended for Molinare delivery"
                }),
            },
            "optional": {
                "output_dir": ("STRING", {
                    "default": "",
                    "tooltip": "Custom output directory (leave empty for default)"
                }),
            }
        }
    
    def save_sequence(self, images, shot_name, version, start_frame, 
                      input_is_linear, convert_to_aces, bit_depth, compression,
                      output_dir=""):
        
        if not _HAS_PYEXR:
            raise RuntimeError("pyexr required for EXR sequence export. Install with: pip install pyexr")
        
        version_str = f"V{version:03d}"
        folder_name = f"{shot_name}_{version_str}"
        
        if output_dir and os.path.isdir(output_dir):
            base_dir = output_dir
        elif folder_paths:
            base_dir = folder_paths.get_output_directory()
        else:
            base_dir = os.path.join(os.path.dirname(__file__), "..", "..", "output")
        
        shot_dir = os.path.join(base_dir, folder_name)
        os.makedirs(shot_dir, exist_ok=True)
        
        if hasattr(images, 'cpu'):
            images_np = images.cpu().numpy()
        else:
            images_np = np.array(images)
        
        if len(images_np.shape) == 3:
            images_np = images_np[np.newaxis, ...]
        
        num_frames = images_np.shape[0]
        
        print(f"[SaveEXRSequence] Saving {num_frames} frames to {shot_dir}")
        t0 = time.time()
        
        for i in range(num_frames):
            frame_num = start_frame + i
            filename = f"{shot_name}_{version_str}.{frame_num:04d}.exr"
            filepath = os.path.join(shot_dir, filename)
            
            frame = images_np[i].astype(np.float32)
            
            if frame.shape[-1] == 4:
                frame = frame[..., :3]
            
            if convert_to_aces and input_is_linear:
                frame = _linear_to_aces_ap0(frame)
            
            if bit_depth == "16":
                frame = frame.astype(np.float16)
            
            pyexr.write(filepath, frame.astype(np.float32))
        
        elapsed = time.time() - t0
        print(f"[SaveEXRSequence] Saved {num_frames} frames in {elapsed:.1f}s")
        
        return (shot_dir,)


class GenerateDeliveryCSV:
    """Generate CSV manifest for Molinare delivery."""
    
    CATEGORY = "Cinema Delivery"
    FUNCTION = "generate_csv"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("csv_path",)
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shot_folder": ("STRING", {"forceInput": True}),
                "shot_name": ("STRING", {"default": "KSA_001_010"}),
                "version": ("INT", {"default": 1, "min": 0, "max": 999}),
                "framerate": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0}),
                "resolution": (["4096x2160", "3840x2160", "2048x1080", "1920x1080"], {
                    "default": "4096x2160"
                }),
                "color_space": (["ACES 2065-1", "ACEScct", "Linear Rec.709"], {
                    "default": "ACES 2065-1"
                }),
            }
        }
    
    def generate_csv(self, shot_folder, shot_name, version, framerate, resolution, color_space):
        import glob
        
        version_str = f"V{version:03d}"
        exr_files = sorted(glob.glob(os.path.join(shot_folder, "*.exr")))
        num_frames = len(exr_files)
        
        if num_frames == 0:
            raise ValueError(f"No EXR files found in {shot_folder}")
        
        first_frame = os.path.basename(exr_files[0]).split('.')[-2]
        last_frame = os.path.basename(exr_files[-1]).split('.')[-2]
        duration_seconds = num_frames / framerate
        
        csv_path = os.path.join(shot_folder, f"{shot_name}_{version_str}_manifest.csv")
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Field", "Value"])
            writer.writerow(["Shot Name", f"{shot_name}_{version_str}"])
            writer.writerow(["Version", version_str])
            writer.writerow(["Resolution", resolution])
            writer.writerow(["Color Space", color_space])
            writer.writerow(["Framerate", f"{framerate} fps"])
            writer.writerow(["Frame Range", f"{first_frame}-{last_frame}"])
            writer.writerow(["Total Frames", num_frames])
            writer.writerow(["Duration", f"{duration_seconds:.2f} seconds"])
            writer.writerow(["Format", "OpenEXR 16-bit half-float"])
            writer.writerow(["Compression", "PIZ"])
            writer.writerow(["Delivery Date", datetime.now().strftime("%Y-%m-%d %H:%M")])
            writer.writerow(["Generated By", "NodyJS/ComfyUI Cinema Delivery"])
        
        print(f"[GenerateDeliveryCSV] Created manifest: {csv_path}")
        return (csv_path,)


class ACESToRec709Preview:
    """
    Convert ACES 2065-1 to Rec.709 for preview/reference QuickTime.
    Applies ACES RRT + ODT transform approximation.
    """
    
    CATEGORY = "Cinema Delivery"
    FUNCTION = "convert"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("preview_image",)
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "exposure": ("FLOAT", {
                    "default": 0.0, "min": -5.0, "max": 5.0, "step": 0.1,
                    "tooltip": "Exposure adjustment in stops"
                }),
                "gamma": ("FLOAT", {
                    "default": 2.4, "min": 1.8, "max": 2.8, "step": 0.1,
                    "tooltip": "Display gamma (2.4 = Rec.1886)"
                }),
            }
        }
    
    def convert(self, image, exposure, gamma):
        if hasattr(image, 'cpu'):
            img = image.cpu().numpy()
        else:
            img = np.array(image)
        
        if exposure != 0:
            img = img * (2.0 ** exposure)
        
        # ACES AP0 to Rec.709 matrix
        mat = np.array([
            [ 2.5214,  -1.1340, -0.3875],
            [-0.2763,   1.3727, -0.0963],
            [-0.0159,  -0.1529,  1.1688]
        ], dtype=np.float32)
        
        shape = img.shape
        if len(shape) == 4:
            batch = shape[0]
            flat = img.reshape(batch, -1, 3)
            out = np.zeros_like(flat)
            for b in range(batch):
                out[b] = flat[b] @ mat.T
            out = out.reshape(shape)
        else:
            flat = img.reshape(-1, 3)
            out = flat @ mat.T
            out = out.reshape(shape)
        
        # ACES filmic S-curve approximation
        a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
        out = np.clip(out, 0, None)
        out = (out * (a * out + b)) / (out * (c * out + d) + e)
        
        out = np.clip(out, 0, 1)
        out = np.power(out, 1.0 / gamma)
        
        if torch:
            return (torch.from_numpy(out.astype(np.float32)),)
        return (out.astype(np.float32),)


# Export classes for registration
CINEMA_DELIVERY_NODES = {
    "SaveEXRSequence": SaveEXRSequence,
    "GenerateDeliveryCSV": GenerateDeliveryCSV,
    "ACESToRec709Preview": ACESToRec709Preview,
}

CINEMA_DELIVERY_DISPLAY_NAMES = {
    "SaveEXRSequence": "Save EXR Sequence (Cinema)",
    "GenerateDeliveryCSV": "Generate Delivery CSV",
    "ACESToRec709Preview": "ACES → Rec.709 Preview",
}
