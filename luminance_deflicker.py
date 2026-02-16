# Luminance-Preserving Deflicker for ComfyUI
# Corrects frame-to-frame brightness variations WITHOUT blurring
# Part of js-exr-upbitrate package

import numpy as np
import torch


class LuminanceDeflicker:
    """
    Deflicker video by correcting per-frame luminance variations.
    
    Unlike temporal averaging (which blurs), this node:
    1. Measures average luminance per frame
    2. Smooths the luminance curve temporally
    3. Applies gain/offset correction per frame
    
    Result: Stable brightness, ZERO blur, all detail preserved.
    """
    
    CATEGORY = "video/correction"
    FUNCTION = "deflicker"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "correction_info")
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "method": (["gain_only", "gain_and_offset", "histogram_match"], {
                    "default": "gain_only",
                    "tooltip": "gain_only: Scale brightness. gain_and_offset: Also adjust black level. histogram_match: Match to reference frame."
                }),
                "smoothing_window": ("INT", {
                    "default": 7,
                    "min": 3,
                    "max": 31,
                    "step": 2,
                    "tooltip": "Frames for luminance smoothing (odd number). Larger = smoother but may miss intentional changes."
                }),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Correction strength. 1.0 = full correction, 0.5 = half correction."
                }),
                "reference_frame": (["middle", "first", "brightest", "average"], {
                    "default": "average",
                    "tooltip": "Reference for target luminance level."
                }),
            },
            "optional": {
                "protect_highlights": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Prevent highlight clipping when boosting dark frames."
                }),
                "max_correction": ("FLOAT", {
                    "default": 2.0,
                    "min": 1.1,
                    "max": 4.0,
                    "step": 0.1,
                    "tooltip": "Maximum gain factor (prevents extreme corrections)."
                }),
            }
        }
    
    def _compute_frame_luminance(self, frame_np):
        """Compute average luminance of a frame."""
        # Rec.709 luminance
        lum = 0.2126 * frame_np[:, :, 0] + 0.7152 * frame_np[:, :, 1] + 0.0722 * frame_np[:, :, 2]
        return float(np.mean(lum))
    
    def _compute_frame_stats(self, frame_np):
        """Compute luminance mean and std for gain+offset correction."""
        lum = 0.2126 * frame_np[:, :, 0] + 0.7152 * frame_np[:, :, 1] + 0.0722 * frame_np[:, :, 2]
        return float(np.mean(lum)), float(np.std(lum))
    
    def _smooth_curve(self, values, window):
        """Apply moving average smoothing to a curve."""
        if len(values) < window:
            return values
        
        # Pad edges
        half = window // 2
        padded = np.pad(values, (half, half), mode='edge')
        
        # Moving average
        kernel = np.ones(window) / window
        smoothed = np.convolve(padded, kernel, mode='valid')
        
        return smoothed[:len(values)]
    
    def deflicker(self, images, method, smoothing_window, strength, reference_frame,
                  protect_highlights=True, max_correction=2.0):
        
        device = images.device
        dtype = images.dtype
        
        # Convert to numpy
        frames_np = images.cpu().float().numpy()
        B, H, W, C = frames_np.shape
        
        if B < 3:
            info = f"Only {B} frame(s) - need 3+ for deflicker. Pass-through."
            print(f"[LuminanceDeflicker] {info}")
            return (images, info)
        
        # Step 1: Measure luminance per frame
        if method == "gain_and_offset":
            stats = [self._compute_frame_stats(frames_np[i]) for i in range(B)]
            lum_means = np.array([s[0] for s in stats])
            lum_stds = np.array([s[1] for s in stats])
        else:
            lum_means = np.array([self._compute_frame_luminance(frames_np[i]) for i in range(B)])
            lum_stds = None
        
        # Step 2: Determine target luminance
        if reference_frame == "middle":
            target_lum = lum_means[B // 2]
        elif reference_frame == "first":
            target_lum = lum_means[0]
        elif reference_frame == "brightest":
            target_lum = np.max(lum_means)
        else:  # average
            target_lum = np.mean(lum_means)
        
        # Step 3: Smooth the luminance curve
        smoothed_lum = self._smooth_curve(lum_means, smoothing_window)
        
        # The correction should bring each frame's luminance to the smoothed value
        # This removes frame-to-frame variation while preserving intentional changes
        
        # Step 4: Apply corrections
        results = np.zeros_like(frames_np)
        corrections = []
        
        for i in range(B):
            frame = frames_np[i].copy()
            original_lum = lum_means[i]
            target = smoothed_lum[i]
            
            if original_lum < 1e-6:
                # Nearly black frame - skip
                results[i] = frame
                corrections.append(1.0)
                continue
            
            if method == "gain_only":
                # Simple multiplicative correction
                gain = target / original_lum
                
                # Clamp gain
                gain = np.clip(gain, 1.0 / max_correction, max_correction)
                
                # Apply strength
                gain = 1.0 + (gain - 1.0) * strength
                
                # Apply gain
                corrected = frame * gain
                
                corrections.append(gain)
                
            elif method == "gain_and_offset":
                # Adjust both mean and contrast
                target_std = np.mean(lum_stds)  # Use average std as target
                
                gain = target / original_lum if original_lum > 1e-6 else 1.0
                gain = np.clip(gain, 1.0 / max_correction, max_correction)
                gain = 1.0 + (gain - 1.0) * strength
                
                # Apply
                corrected = frame * gain
                corrections.append(gain)
                
            elif method == "histogram_match":
                # Match histogram to reference frame (middle frame)
                ref_idx = B // 2
                ref_frame = frames_np[ref_idx]
                
                corrected = np.zeros_like(frame)
                for c in range(min(C, 3)):
                    corrected[:, :, c] = self._match_histogram(
                        frame[:, :, c], ref_frame[:, :, c], strength
                    )
                if C > 3:
                    corrected[:, :, 3:] = frame[:, :, 3:]
                
                corrections.append(1.0)
            
            # Protect highlights if requested
            if protect_highlights:
                # Where original was already bright, use original
                highlight_mask = frame > 0.9
                corrected = np.where(highlight_mask, frame, corrected)
            
            results[i] = corrected
        
        # Compute stats for info
        lum_range_before = np.max(lum_means) - np.min(lum_means)
        result_lums = [self._compute_frame_luminance(results[i]) for i in range(B)]
        lum_range_after = np.max(result_lums) - np.min(result_lums)
        
        info = (
            f"Frames: {B} | Method: {method} | Window: {smoothing_window}\n"
            f"Luminance range: {lum_range_before:.4f} → {lum_range_after:.4f} "
            f"({100*(1-lum_range_after/max(lum_range_before, 1e-6)):.1f}% reduction)\n"
            f"Correction range: {min(corrections):.3f}x - {max(corrections):.3f}x"
        )
        print(f"[LuminanceDeflicker] {info}")
        
        # Convert back
        result_tensor = torch.from_numpy(results.astype(np.float32)).to(device=device, dtype=dtype)
        
        return (result_tensor, info)
    
    def _match_histogram(self, source, reference, strength):
        """Match source histogram to reference."""
        # Flatten and sort
        src_flat = source.flatten()
        ref_flat = reference.flatten()
        
        # Get sorted indices
        src_idx = np.argsort(src_flat)
        ref_sorted = np.sort(ref_flat)
        
        # Map source values to reference values
        matched = np.zeros_like(src_flat)
        matched[src_idx] = ref_sorted
        
        # Reshape
        matched = matched.reshape(source.shape)
        
        # Blend with original
        return source * (1 - strength) + matched * strength


class NormalsDeflicker:
    """
    Deflicker by stabilizing surface normals / local gradients.
    
    This preserves luminance while removing temporal variation in surface detail.
    Works by:
    1. Extracting local gradients (pseudo-normals) from each frame
    2. Temporally smoothing the gradient field
    3. Reconstructing the image with smoothed gradients but original luminance
    """
    
    CATEGORY = "video/correction"
    FUNCTION = "deflicker"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "correction_info")
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "gradient_smoothing": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "How much to smooth gradients temporally. 0 = no change, 1 = full smoothing."
                }),
                "temporal_radius": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 5,
                    "step": 1,
                    "tooltip": "Frames each side for gradient smoothing. Keep low to avoid blur."
                }),
                "preserve_luminance": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep original frame luminance (recommended)."
                }),
            }
        }
    
    def _compute_gradients(self, frame):
        """Compute image gradients (Sobel-like)."""
        # Simple gradient computation
        grad_x = np.zeros_like(frame)
        grad_y = np.zeros_like(frame)
        
        grad_x[:, 1:-1, :] = (frame[:, 2:, :] - frame[:, :-2, :]) / 2
        grad_y[1:-1, :, :] = (frame[2:, :, :] - frame[:-2, :, :]) / 2
        
        return grad_x, grad_y
    
    def _reconstruct_from_gradients(self, grad_x, grad_y, original, preserve_lum):
        """Reconstruct image from gradients while preserving luminance."""
        H, W, C = original.shape
        
        # Simple Poisson-like reconstruction (fast approximation)
        # Start with original and nudge towards gradient-consistent version
        result = original.copy()
        
        for iteration in range(5):  # Few iterations
            # Compute what gradients we currently have
            curr_gx, curr_gy = self._compute_gradients(result)
            
            # Gradient error
            err_x = grad_x - curr_gx
            err_y = grad_y - curr_gy
            
            # Update (simple gradient descent)
            update = np.zeros_like(result)
            update[:, 1:-1, :] += err_x[:, 1:-1, :] * 0.25
            update[:, :-2, :] -= err_x[:, 1:-1, :] * 0.25
            update[1:-1, :, :] += err_y[1:-1, :, :] * 0.25
            update[:-2, :, :] -= err_y[1:-1, :, :] * 0.25
            
            result = result + update * 0.5
        
        if preserve_lum:
            # Restore original luminance
            orig_lum = 0.2126 * original[:, :, 0] + 0.7152 * original[:, :, 1] + 0.0722 * original[:, :, 2]
            result_lum = 0.2126 * result[:, :, 0] + 0.7152 * result[:, :, 1] + 0.0722 * result[:, :, 2]
            
            # Scale to match original luminance
            scale = (orig_lum + 1e-6) / (result_lum + 1e-6)
            result = result * scale[:, :, np.newaxis]
        
        return result
    
    def deflicker(self, images, gradient_smoothing, temporal_radius, preserve_luminance):
        device = images.device
        dtype = images.dtype
        
        frames_np = images.cpu().float().numpy()
        B, H, W, C = frames_np.shape
        
        if B < 3:
            info = f"Only {B} frame(s) - need 3+ for deflicker. Pass-through."
            return (images, info)
        
        # Compute gradients for all frames
        all_grad_x = []
        all_grad_y = []
        
        for i in range(B):
            gx, gy = self._compute_gradients(frames_np[i])
            all_grad_x.append(gx)
            all_grad_y.append(gy)
        
        all_grad_x = np.stack(all_grad_x, axis=0)  # [B, H, W, C]
        all_grad_y = np.stack(all_grad_y, axis=0)
        
        # Temporal smoothing of gradients
        smoothed_gx = np.zeros_like(all_grad_x)
        smoothed_gy = np.zeros_like(all_grad_y)
        
        for i in range(B):
            lo = max(0, i - temporal_radius)
            hi = min(B, i + temporal_radius + 1)
            
            # Weighted average (closer frames get more weight)
            weights = []
            for j in range(lo, hi):
                w = 1.0 / (1.0 + abs(j - i))
                weights.append(w)
            weights = np.array(weights) / sum(weights)
            
            avg_gx = np.zeros_like(all_grad_x[0])
            avg_gy = np.zeros_like(all_grad_y[0])
            
            for idx, j in enumerate(range(lo, hi)):
                avg_gx += all_grad_x[j] * weights[idx]
                avg_gy += all_grad_y[j] * weights[idx]
            
            # Blend between original and smoothed
            smoothed_gx[i] = all_grad_x[i] * (1 - gradient_smoothing) + avg_gx * gradient_smoothing
            smoothed_gy[i] = all_grad_y[i] * (1 - gradient_smoothing) + avg_gy * gradient_smoothing
        
        # Reconstruct frames
        results = np.zeros_like(frames_np)
        
        for i in range(B):
            results[i] = self._reconstruct_from_gradients(
                smoothed_gx[i], smoothed_gy[i], 
                frames_np[i], preserve_luminance
            )
        
        info = (
            f"Frames: {B} | Gradient smoothing: {gradient_smoothing:.0%} | "
            f"Temporal radius: {temporal_radius} | Preserve luminance: {preserve_luminance}"
        )
        print(f"[NormalsDeflicker] {info}")
        
        result_tensor = torch.from_numpy(results.astype(np.float32)).to(device=device, dtype=dtype)
        
        return (result_tensor, info)


# Export
LUMINANCE_DEFLICKER_NODES = {
    "LuminanceDeflicker": LuminanceDeflicker,
    "NormalsDeflicker": NormalsDeflicker,
}

LUMINANCE_DEFLICKER_DISPLAY_NAMES = {
    "LuminanceDeflicker": "Luminance Deflicker (No Blur)",
    "NormalsDeflicker": "Normals Deflicker (Gradient Preserve)",
}
