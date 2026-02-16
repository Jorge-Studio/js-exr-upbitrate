# Animated Motion Control Nodes for ComfyUI
# Provides keyframe-based animation for pan, zoom, and camera movements.
# Part of js-exr-upbitrate package

import numpy as np
import math

try:
    import torch
except ImportError:
    torch = None


def ease_in_out(t: float) -> float:
    """Smooth ease-in-out interpolation."""
    return t * t * (3 - 2 * t)


def ease_in(t: float) -> float:
    """Ease-in interpolation."""
    return t * t


def ease_out(t: float) -> float:
    """Ease-out interpolation."""
    return 1 - (1 - t) * (1 - t)


def linear(t: float) -> float:
    """Linear interpolation."""
    return t


EASING_FUNCTIONS = {
    "linear": linear,
    "ease_in": ease_in,
    "ease_out": ease_out,
    "ease_in_out": ease_in_out,
}


class AnimatedPanAndScan:
    """
    Animate pan and scan over a video sequence.
    Define start and end positions, and the node interpolates across all frames.
    Perfect for creating camera movements across large images or video sequences.
    """
    
    CATEGORY = "animation/motion"
    FUNCTION = "animate"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("animated_sequence",)
    
    OUTPUT_SIZES = {
        "4K DCI (4096×2160)": (4096, 2160),
        "4K UHD (3840×2160)": (3840, 2160),
        "2K DCI (2048×1080)": (2048, 1080),
        "1080p (1920×1080)": (1920, 1080),
        "720p (1280×720)": (1280, 720),
        "Custom": (0, 0),
    }
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "output_size": (list(cls.OUTPUT_SIZES.keys()), {"default": "4K DCI (4096×2160)"}),
                "custom_width": ("INT", {"default": 4096, "min": 64, "max": 16384, "step": 8}),
                "custom_height": ("INT", {"default": 2160, "min": 64, "max": 16384, "step": 8}),
                "start_x": ("INT", {"default": 0, "min": -16384, "max": 16384, "step": 1,
                    "tooltip": "Starting X offset (pan position)"}),
                "start_y": ("INT", {"default": 0, "min": -16384, "max": 16384, "step": 1,
                    "tooltip": "Starting Y offset (tilt position)"}),
                "start_zoom": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.01,
                    "tooltip": "Starting zoom level (1.0 = 100%)"}),
                "end_x": ("INT", {"default": 0, "min": -16384, "max": 16384, "step": 1,
                    "tooltip": "Ending X offset"}),
                "end_y": ("INT", {"default": 0, "min": -16384, "max": 16384, "step": 1,
                    "tooltip": "Ending Y offset"}),
                "end_zoom": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.01,
                    "tooltip": "Ending zoom level"}),
                "easing": (["linear", "ease_in", "ease_out", "ease_in_out"], {
                    "default": "ease_in_out", "tooltip": "Interpolation curve type"}),
                "loop_mode": (["none", "ping_pong", "repeat"], {
                    "default": "none", "tooltip": "How to handle animation beyond keyframes"}),
            },
            "optional": {
                "start_frame": ("INT", {"default": 0, "min": 0, "max": 99999,
                    "tooltip": "Frame to start animation (0 = first frame)"}),
                "end_frame": ("INT", {"default": -1, "min": -1, "max": 99999,
                    "tooltip": "Frame to end animation (-1 = last frame)"}),
                "rotation_start": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.5,
                    "tooltip": "Starting rotation in degrees"}),
                "rotation_end": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.5,
                    "tooltip": "Ending rotation in degrees"}),
            }
        }
    
    def animate(self, images, output_size, custom_width, custom_height,
                start_x, start_y, start_zoom, end_x, end_y, end_zoom,
                easing, loop_mode, start_frame=0, end_frame=-1,
                rotation_start=0.0, rotation_end=0.0):
        
        if output_size == "Custom":
            out_w, out_h = custom_width, custom_height
        else:
            out_w, out_h = self.OUTPUT_SIZES[output_size]
        
        if hasattr(images, 'cpu'):
            images_np = images.cpu().numpy()
        else:
            images_np = np.array(images)
        
        if len(images_np.shape) == 3:
            images_np = images_np[np.newaxis, ...]
        
        num_frames = images_np.shape[0]
        _, src_h, src_w, channels = images_np.shape
        
        if end_frame < 0:
            end_frame = num_frames - 1
        end_frame = min(end_frame, num_frames - 1)
        
        ease_fn = EASING_FUNCTIONS.get(easing, linear)
        output_frames = []
        
        for frame_idx in range(num_frames):
            if end_frame > start_frame:
                t = (frame_idx - start_frame) / (end_frame - start_frame)
            else:
                t = 0.0
            
            if loop_mode == "none":
                t = max(0.0, min(1.0, t))
            elif loop_mode == "ping_pong":
                t = t % 2.0
                if t > 1.0:
                    t = 2.0 - t
            elif loop_mode == "repeat":
                t = t % 1.0
            
            t_eased = ease_fn(t)
            
            current_x = start_x + (end_x - start_x) * t_eased
            current_y = start_y + (end_y - start_y) * t_eased
            current_zoom = start_zoom + (end_zoom - start_zoom) * t_eased
            current_rotation = rotation_start + (rotation_end - rotation_start) * t_eased
            
            src_frame = images_np[frame_idx]
            
            crop_w = int(out_w / current_zoom)
            crop_h = int(out_h / current_zoom)
            crop_w = min(crop_w, src_w)
            crop_h = min(crop_h, src_h)
            
            crop_x = int((src_w - crop_w) / 2 + current_x)
            crop_y = int((src_h - crop_h) / 2 + current_y)
            crop_x = max(0, min(crop_x, src_w - crop_w))
            crop_y = max(0, min(crop_y, src_h - crop_h))
            
            cropped = src_frame[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w, :]
            
            if abs(current_rotation) > 0.1:
                import cv2
                center = (cropped.shape[1] // 2, cropped.shape[0] // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, current_rotation, 1.0)
                cropped = cv2.warpAffine(cropped, rotation_matrix, 
                    (cropped.shape[1], cropped.shape[0]),
                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            
            from PIL import Image
            pil_img = Image.fromarray((cropped * 255).astype(np.uint8))
            pil_img = pil_img.resize((out_w, out_h), Image.LANCZOS)
            resized = np.array(pil_img).astype(np.float32) / 255.0
            
            output_frames.append(resized)
        
        output = np.stack(output_frames, axis=0)
        
        if torch:
            return (torch.from_numpy(output.astype(np.float32)),)
        return (output.astype(np.float32),)


class LoadEXRImage:
    """
    Load a single EXR image file.
    Handles HDR EXR files that standard ComfyUI LoadImage cannot read.
    """
    
    CATEGORY = "loaders"
    FUNCTION = "load"
    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "width", "height")
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {
                    "default": "",
                    "tooltip": "Full path to EXR file"
                }),
            }
        }
    
    def load(self, file_path):
        import os
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"EXR file not found: {file_path}")
        
        try:
            import pyexr
            img = pyexr.read(file_path)
        except ImportError:
            try:
                import imageio
                img = imageio.imread(file_path)
            except Exception as e:
                raise RuntimeError(f"Cannot load EXR: {e}. Install pyexr or imageio.")
        
        if img.dtype != np.float32:
            img = img.astype(np.float32)
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=-1)
        if img.shape[-1] == 4:
            img = img[..., :3]
        
        height, width = img.shape[:2]
        img_batch = img[np.newaxis, ...]
        
        print(f"[LoadEXRImage] Loaded {width}x{height} EXR, range [{img.min():.4f}, {img.max():.4f}]")
        
        if torch:
            return (torch.from_numpy(img_batch), width, height)
        return (img_batch, width, height)


class LoadEXRSequence:
    """
    Load an EXR image sequence as a video batch.
    Supports reading EXR sequences for processing through the pipeline.
    """
    
    CATEGORY = "loaders/video"
    FUNCTION = "load"
    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT")
    RETURN_NAMES = ("images", "frame_count", "width", "height")
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to folder containing EXR sequence"
                }),
                "start_frame": ("INT", {
                    "default": 1001, "min": 0, "max": 999999,
                    "tooltip": "First frame number to load (0 = load all)"
                }),
                "frame_count": ("INT", {
                    "default": 120, "min": 1, "max": 9999,
                    "tooltip": "Number of frames to load"
                }),
                "skip_frames": ("INT", {
                    "default": 1, "min": 1, "max": 100,
                    "tooltip": "Load every Nth frame (1 = all frames)"
                }),
            }
        }
    
    def load(self, folder_path, start_frame, frame_count, skip_frames):
        import glob
        import os
        import re
        
        exr_pattern = os.path.join(folder_path, "*.exr")
        exr_files = sorted(glob.glob(exr_pattern))
        
        if not exr_files:
            raise ValueError(f"No EXR files found in {folder_path}")
        
        def get_frame_num(filepath):
            name = os.path.basename(filepath)
            matches = re.findall(r'(\d+)', name)
            if matches:
                return int(matches[-1])
            return 0
        
        exr_files = sorted(exr_files, key=get_frame_num)
        
        frames_to_load = []
        if start_frame == 0:
            frames_to_load = exr_files[:frame_count] if frame_count > 0 else exr_files
        else:
            for f in exr_files:
                num = get_frame_num(f)
                if num >= start_frame:
                    frames_to_load.append(f)
                    if len(frames_to_load) >= frame_count and frame_count > 0:
                        break
        
        frames_to_load = frames_to_load[::skip_frames]
        
        if not frames_to_load:
            print(f"[LoadEXRSequence] Warning: No frames matched filter, loading all {len(exr_files)} files")
            frames_to_load = exr_files[:frame_count] if frame_count > 0 else exr_files
            frames_to_load = frames_to_load[::skip_frames]
        
        if not frames_to_load:
            raise ValueError(f"No EXR files found in {folder_path}")
        
        try:
            import pyexr
            use_pyexr = True
        except ImportError:
            use_pyexr = False
        
        loaded_frames = []
        for filepath in frames_to_load:
            if use_pyexr:
                import pyexr
                img = pyexr.read(filepath)
            else:
                import imageio
                img = imageio.imread(filepath)
            
            if img.dtype != np.float32:
                img = img.astype(np.float32)
            if len(img.shape) == 2:
                img = np.stack([img, img, img], axis=-1)
            if img.shape[-1] == 4:
                img = img[..., :3]
            
            loaded_frames.append(img)
        
        batch = np.stack(loaded_frames, axis=0)
        height, width = batch.shape[1], batch.shape[2]
        
        print(f"[LoadEXRSequence] Loaded {len(loaded_frames)} frames, {width}x{height}")
        
        if torch:
            return (torch.from_numpy(batch), len(loaded_frames), width, height)
        return (batch, len(loaded_frames), width, height)


class MotionPathFromVideo:
    """
    Extract motion/camera movement from a reference video.
    Uses optical flow to detect motion patterns.
    """
    
    CATEGORY = "animation/motion"
    FUNCTION = "extract"
    RETURN_TYPES = ("MOTION_PATH",)
    RETURN_NAMES = ("motion_path",)
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_video": ("IMAGE",),
                "sensitivity": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1,
                    "tooltip": "Motion detection sensitivity"
                }),
                "smoothing": ("INT", {
                    "default": 3, "min": 1, "max": 15, "step": 2,
                    "tooltip": "Temporal smoothing window"
                }),
            }
        }
    
    def extract(self, reference_video, sensitivity, smoothing):
        import cv2
        
        if hasattr(reference_video, 'cpu'):
            video = reference_video.cpu().numpy()
        else:
            video = np.array(reference_video)
        
        if len(video.shape) == 3:
            video = video[np.newaxis, ...]
        
        num_frames = video.shape[0]
        motion_data = {
            "x_offsets": [], "y_offsets": [],
            "rotations": [], "zooms": [],
            "num_frames": num_frames,
        }
        
        if num_frames < 2:
            motion_data["x_offsets"] = [0.0]
            motion_data["y_offsets"] = [0.0]
            motion_data["rotations"] = [0.0]
            motion_data["zooms"] = [1.0]
            return (motion_data,)
        
        prev_gray = cv2.cvtColor((video[0] * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        cumulative_x = 0.0
        cumulative_y = 0.0
        
        for i in range(num_frames):
            if i == 0:
                motion_data["x_offsets"].append(0.0)
                motion_data["y_offsets"].append(0.0)
                motion_data["rotations"].append(0.0)
                motion_data["zooms"].append(1.0)
                continue
            
            curr_gray = cv2.cvtColor((video[i] * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            
            avg_x = np.mean(flow[..., 0]) * sensitivity
            avg_y = np.mean(flow[..., 1]) * sensitivity
            
            cumulative_x += avg_x
            cumulative_y += avg_y
            
            motion_data["x_offsets"].append(float(cumulative_x))
            motion_data["y_offsets"].append(float(cumulative_y))
            motion_data["rotations"].append(0.0)
            motion_data["zooms"].append(1.0)
            
            prev_gray = curr_gray
        
        if smoothing > 1:
            from scipy.ndimage import uniform_filter1d
            motion_data["x_offsets"] = uniform_filter1d(
                np.array(motion_data["x_offsets"]), size=smoothing
            ).tolist()
            motion_data["y_offsets"] = uniform_filter1d(
                np.array(motion_data["y_offsets"]), size=smoothing
            ).tolist()
        
        print(f"[MotionPathFromVideo] Extracted motion from {num_frames} frames")
        
        return (motion_data,)


class ApplyMotionPath:
    """
    Apply a motion path to an image/video sequence.
    Uses extracted or manually defined motion paths.
    """
    
    CATEGORY = "animation/motion"
    FUNCTION = "apply"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("animated_sequence",)
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "motion_path": ("MOTION_PATH",),
                "output_width": ("INT", {"default": 4096, "min": 64, "max": 16384, "step": 8}),
                "output_height": ("INT", {"default": 2160, "min": 64, "max": 16384, "step": 8}),
                "scale_motion": ("FLOAT", {
                    "default": 1.0, "min": -10.0, "max": 10.0, "step": 0.1,
                    "tooltip": "Scale the motion (negative = reverse direction)"
                }),
                "invert_motion": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Invert motion direction (stabilization mode)"
                }),
            }
        }
    
    def apply(self, images, motion_path, output_width, output_height, scale_motion, invert_motion):
        from PIL import Image
        
        if hasattr(images, 'cpu'):
            images_np = images.cpu().numpy()
        else:
            images_np = np.array(images)
        
        if len(images_np.shape) == 3:
            images_np = images_np[np.newaxis, ...]
        
        num_frames = images_np.shape[0]
        _, src_h, src_w, channels = images_np.shape
        
        x_offsets = motion_path["x_offsets"]
        y_offsets = motion_path["y_offsets"]
        
        while len(x_offsets) < num_frames:
            x_offsets.append(x_offsets[-1])
            y_offsets.append(y_offsets[-1])
        
        output_frames = []
        
        for i in range(num_frames):
            offset_x = x_offsets[min(i, len(x_offsets)-1)] * scale_motion
            offset_y = y_offsets[min(i, len(y_offsets)-1)] * scale_motion
            
            if invert_motion:
                offset_x = -offset_x
                offset_y = -offset_y
            
            crop_w = min(output_width, src_w)
            crop_h = min(output_height, src_h)
            
            crop_x = int((src_w - crop_w) / 2 + offset_x)
            crop_y = int((src_h - crop_h) / 2 + offset_y)
            crop_x = max(0, min(crop_x, src_w - crop_w))
            crop_y = max(0, min(crop_y, src_h - crop_h))
            
            cropped = images_np[i, crop_y:crop_y+crop_h, crop_x:crop_x+crop_w, :]
            
            pil_img = Image.fromarray((cropped * 255).astype(np.uint8))
            pil_img = pil_img.resize((output_width, output_height), Image.LANCZOS)
            resized = np.array(pil_img).astype(np.float32) / 255.0
            
            output_frames.append(resized)
        
        output = np.stack(output_frames, axis=0)
        
        if torch:
            return (torch.from_numpy(output.astype(np.float32)),)
        return (output.astype(np.float32),)


# Export classes for registration
ANIMATED_MOTION_NODES = {
    "AnimatedPanAndScan": AnimatedPanAndScan,
    "LoadEXRImage": LoadEXRImage,
    "LoadEXRSequence": LoadEXRSequence,
    "MotionPathFromVideo": MotionPathFromVideo,
    "ApplyMotionPath": ApplyMotionPath,
}

ANIMATED_MOTION_DISPLAY_NAMES = {
    "AnimatedPanAndScan": "Animated Pan & Scan",
    "LoadEXRImage": "Load EXR Image",
    "LoadEXRSequence": "Load EXR Sequence",
    "MotionPathFromVideo": "Extract Motion from Video",
    "ApplyMotionPath": "Apply Motion Path",
}
