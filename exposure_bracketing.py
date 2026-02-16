# Exposure Bracketing for ComfyUI
# Generate multiple exposure brackets from a single source for HDR merging
# Part of js-exr-upbitrate package

import os
import numpy as np

try:
    import torch
except ImportError:
    torch = None

try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False


class ExposureBracketGenerator:
    """
    Generate 5 exposure brackets from a single image/video source.
    
    Takes EXR, video frames, or any image input and creates:
    - EV +4 (4 stops overexposed)
    - EV +2 (2 stops overexposed)  
    - EV 0 (original)
    - EV -2 (2 stops underexposed)
    - EV -4 (4 stops underexposed)
    
    These can then be fed into DeterministicHDRMerge5 for bit-depth expansion.
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
                "method": (["linear", "gamma_aware", "filmic"], {
                    "default": "gamma_aware",
                    "tooltip": "How to apply exposure changes"
                }),
                "clip_highlights": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Clip overexposed values to 1.0 (simulates real camera)"
                }),
                "add_noise_to_darks": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Add realistic sensor noise to underexposed brackets"
                }),
            },
            "optional": {
                "noise_amount": ("FLOAT", {
                    "default": 0.01,
                    "min": 0.0,
                    "max": 0.1,
                    "step": 0.005,
                    "tooltip": "Amount of noise for dark brackets"
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 2**31 - 1,
                    "tooltip": "Random seed for noise"
                }),
            }
        }
    
    def _apply_exposure(self, img, stops, method, clip):
        """Apply exposure change in stops."""
        multiplier = 2.0 ** stops
        
        if method == "linear":
            result = img * multiplier
        elif method == "gamma_aware":
            # Work in linear space for accurate exposure
            linear = np.power(np.clip(img, 0, 1), 2.2)
            exposed = linear * multiplier
            result = np.power(np.clip(exposed, 0, None), 1/2.2)
        elif method == "filmic":
            # Filmic response - softer highlights
            linear = np.power(np.clip(img, 0, 1), 2.2)
            exposed = linear * multiplier
            # Soft shoulder
            result = exposed / (exposed + 1) * 2
            result = np.power(np.clip(result, 0, 1), 1/2.2)
        else:
            result = img * multiplier
        
        if clip:
            result = np.clip(result, 0, 1)
        
        return result.astype(np.float32)
    
    def _add_noise(self, img, amount, seed, stops):
        """Add noise that increases with underexposure."""
        np.random.seed(seed)
        
        # More noise for darker brackets
        noise_scale = amount * (1 + abs(min(stops, 0)) * 0.5)
        
        noise = np.random.normal(0, noise_scale, img.shape).astype(np.float32)
        
        # Apply noise more to shadows
        shadow_mask = 1 - np.clip(img, 0, 1)
        noisy = img + noise * shadow_mask
        
        return np.clip(noisy, 0, 1).astype(np.float32)
    
    def generate_brackets(self, image, exposure_step, method, clip_highlights, 
                          add_noise_to_darks, noise_amount=0.01, seed=42):
        
        if hasattr(image, 'cpu'):
            img_np = image.cpu().float().numpy()
        else:
            img_np = np.array(image, dtype=np.float32)
        
        # Generate 5 brackets
        stops = [exposure_step * 2, exposure_step, 0, -exposure_step, -exposure_step * 2]
        brackets = []
        
        for i, ev in enumerate(stops):
            bracket = self._apply_exposure(img_np, ev, method, clip_highlights)
            
            # Add noise to underexposed brackets
            if add_noise_to_darks and ev < 0:
                bracket = self._add_noise(bracket, noise_amount, seed + i, ev)
            
            if torch:
                brackets.append(torch.from_numpy(bracket))
            else:
                brackets.append(bracket)
        
        print(f"[ExposureBracketGenerator] Generated 5 brackets: "
              f"+{stops[0]:.1f}, +{stops[1]:.1f}, {stops[2]:.1f}, {stops[3]:.1f}, {stops[4]:.1f} EV")
        
        return tuple(brackets)


class ExposureBracketToTIFF:
    """
    Save exposure brackets as TIFF sequences for external processing or archival.
    Creates folder structure matching the bit-depth workflow expectations.
    """
    
    CATEGORY = "image/hdr"
    FUNCTION = "save_brackets"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("path_plus_4", "path_plus_2", "path_0", "path_minus_2", "path_minus_4")
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ev_plus_4": ("IMAGE",),
                "ev_plus_2": ("IMAGE",),
                "ev_0": ("IMAGE",),
                "ev_minus_2": ("IMAGE",),
                "ev_minus_4": ("IMAGE",),
                "output_folder": ("STRING", {
                    "default": "",
                    "tooltip": "Base folder for TIFF output"
                }),
                "shot_name": ("STRING", {
                    "default": "shot",
                    "tooltip": "Shot name for folder naming"
                }),
                "bit_depth": (["8", "16"], {
                    "default": "16",
                    "tooltip": "TIFF bit depth"
                }),
                "start_frame": ("INT", {
                    "default": 1001,
                    "min": 0,
                    "max": 999999,
                    "tooltip": "Starting frame number"
                }),
            }
        }
    
    def save_brackets(self, ev_plus_4, ev_plus_2, ev_0, ev_minus_2, ev_minus_4,
                      output_folder, shot_name, bit_depth, start_frame):
        
        if not _HAS_PIL:
            raise RuntimeError("Pillow required for TIFF export. Install with: pip install pillow")
        
        # Create folder structure
        if not output_folder:
            import folder_paths
            output_folder = folder_paths.get_output_directory()
        
        bracket_names = ["+4", "+2", "0", "-2", "-4"]
        bracket_images = [ev_plus_4, ev_plus_2, ev_0, ev_minus_2, ev_minus_4]
        output_paths = []
        
        for name, images in zip(bracket_names, bracket_images):
            folder_name = f"{shot_name}_{name.replace('+', 'plus').replace('-', 'minus')}"
            folder_path = os.path.join(output_folder, folder_name)
            os.makedirs(folder_path, exist_ok=True)
            
            if hasattr(images, 'cpu'):
                images_np = images.cpu().float().numpy()
            else:
                images_np = np.array(images, dtype=np.float32)
            
            if len(images_np.shape) == 3:
                images_np = images_np[np.newaxis, ...]
            
            num_frames = images_np.shape[0]
            
            for i in range(num_frames):
                frame_num = start_frame + i
                filename = f"{shot_name}_{name.replace('+', 'plus').replace('-', 'minus')}_{frame_num:04d}.tiff"
                filepath = os.path.join(folder_path, filename)
                
                frame = images_np[i]
                
                if bit_depth == "16":
                    # 16-bit TIFF
                    frame_16 = (np.clip(frame, 0, 1) * 65535).astype(np.uint16)
                    img = Image.fromarray(frame_16, mode='RGB')
                else:
                    # 8-bit TIFF
                    frame_8 = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
                    img = Image.fromarray(frame_8, mode='RGB')
                
                img.save(filepath, format='TIFF', compression='none')
            
            output_paths.append(folder_path)
            print(f"[ExposureBracketToTIFF] Saved {num_frames} frames to {folder_path}")
        
        return tuple(output_paths)


class VideoToExposureBrackets:
    """
    Load video/EXR and generate exposure brackets in one node.
    Complete solution for bit-depth expansion from a single source.
    """
    
    CATEGORY = "image/hdr"
    FUNCTION = "process"
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("ev_plus_4", "ev_plus_2", "ev_0", "ev_minus_2", "ev_minus_4", "original")
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to video file or EXR sequence folder"
                }),
                "source_type": (["video", "exr_sequence", "image_sequence"], {
                    "default": "video",
                    "tooltip": "Type of source media"
                }),
                "frame_limit": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 9999,
                    "tooltip": "Max frames to load (0 = all)"
                }),
                "exposure_step": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.5,
                    "max": 4.0,
                    "step": 0.5,
                    "tooltip": "Exposure step in stops"
                }),
                "method": (["linear", "gamma_aware", "filmic"], {
                    "default": "gamma_aware"
                }),
            },
            "optional": {
                "start_frame": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 99999
                }),
            }
        }
    
    def process(self, source_path, source_type, frame_limit, exposure_step, method, start_frame=0):
        import glob
        
        frames = []
        
        if source_type == "video":
            # Load video frames
            try:
                import cv2
                cap = cv2.VideoCapture(source_path)
                
                if start_frame > 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                
                count = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # BGR to RGB, normalize
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                    frames.append(frame)
                    
                    count += 1
                    if frame_limit > 0 and count >= frame_limit:
                        break
                
                cap.release()
            except Exception as e:
                raise RuntimeError(f"Failed to load video: {e}")
        
        elif source_type == "exr_sequence":
            # Load EXR sequence
            try:
                import pyexr
                use_pyexr = True
            except ImportError:
                use_pyexr = False
            
            exr_files = sorted(glob.glob(os.path.join(source_path, "*.exr")))
            
            if start_frame > 0:
                exr_files = exr_files[start_frame:]
            if frame_limit > 0:
                exr_files = exr_files[:frame_limit]
            
            for filepath in exr_files:
                if use_pyexr:
                    import pyexr
                    frame = pyexr.read(filepath).astype(np.float32)
                else:
                    import imageio
                    frame = imageio.imread(filepath).astype(np.float32)
                
                if frame.shape[-1] == 4:
                    frame = frame[..., :3]
                
                frames.append(frame)
        
        elif source_type == "image_sequence":
            # Load image sequence (TIFF, PNG, etc.)
            patterns = ["*.tiff", "*.tif", "*.png", "*.jpg", "*.jpeg"]
            all_files = []
            for pattern in patterns:
                all_files.extend(glob.glob(os.path.join(source_path, pattern)))
            
            image_files = sorted(all_files)
            
            if start_frame > 0:
                image_files = image_files[start_frame:]
            if frame_limit > 0:
                image_files = image_files[:frame_limit]
            
            for filepath in image_files:
                img = Image.open(filepath)
                frame = np.array(img).astype(np.float32)
                
                # Normalize based on bit depth
                if frame.max() > 1:
                    if frame.max() > 255:
                        frame = frame / 65535.0  # 16-bit
                    else:
                        frame = frame / 255.0   # 8-bit
                
                if len(frame.shape) == 2:
                    frame = np.stack([frame, frame, frame], axis=-1)
                if frame.shape[-1] == 4:
                    frame = frame[..., :3]
                
                frames.append(frame)
        
        if not frames:
            raise ValueError(f"No frames loaded from {source_path}")
        
        # Stack frames
        original = np.stack(frames, axis=0).astype(np.float32)
        
        print(f"[VideoToExposureBrackets] Loaded {len(frames)} frames from {source_path}")
        
        # Generate brackets
        bracket_gen = ExposureBracketGenerator()
        
        if torch:
            original_tensor = torch.from_numpy(original)
        else:
            original_tensor = original
        
        ev_p4, ev_p2, ev_0, ev_m2, ev_m4 = bracket_gen.generate_brackets(
            original_tensor, exposure_step, method, 
            clip_highlights=True, add_noise_to_darks=True,
            noise_amount=0.01, seed=42
        )
        
        return (ev_p4, ev_p2, ev_0, ev_m2, ev_m4, original_tensor)


# Export classes
EXPOSURE_BRACKETING_NODES = {
    "ExposureBracketGenerator": ExposureBracketGenerator,
    "ExposureBracketToTIFF": ExposureBracketToTIFF,
    "VideoToExposureBrackets": VideoToExposureBrackets,
}

EXPOSURE_BRACKETING_DISPLAY_NAMES = {
    "ExposureBracketGenerator": "Generate Exposure Brackets",
    "ExposureBracketToTIFF": "Save Brackets as TIFF",
    "VideoToExposureBrackets": "Video → Exposure Brackets",
}
