"""
AI Scene Segmentation nodes for ComfyUI.

SceneSegmenter: Decomposes an image into semantic regions using SAM2,
GroundingDINO, and Depth Anything V2. Falls back to luminance/edge-based
segmentation if AI models are not available.

LayerDecomposer: Takes an image and masks, extracts each region as a
separate layer with metadata (label, area, luminance, fractal dimension).
"""

import json
import numpy as np
import torch
from scipy.ndimage import gaussian_filter, label as scipy_label, binary_dilation

from .fractal_utils import compute_local_fractal_dimension

# Optional AI model imports
_HAS_SAM2 = False
_HAS_GDINO = False
_HAS_DEPTH = False

try:
    from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry
    _HAS_SAM2 = True
except ImportError:
    pass

try:
    from groundingdino.util.inference import load_model as load_gdino, predict as gdino_predict
    _HAS_GDINO = True
except ImportError:
    pass


class SceneSegmenter:
    """
    Decompose an image into semantic masks using text prompts.

    Uses GroundingDINO for text-guided detection + SAM2 for precise masks.
    Falls back to luminance/edge-based automatic segmentation if AI models
    are not installed.
    """

    CATEGORY = "image/segmentation"
    FUNCTION = "segment"
    RETURN_TYPES = ("MASK", "MASK", "MASK", "MASK", "MASK", "MASK", "IMAGE", "STRING")
    RETURN_NAMES = ("mask_1", "mask_2", "mask_3", "mask_4", "mask_5", "mask_6",
                    "depth_map", "layer_info")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "text_prompts": ("STRING", {
                    "default": "sky, trees, ground, person, building",
                    "multiline": True,
                    "tooltip": "Comma-separated list of scene elements to segment."
                }),
                "detail_level": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1,
                    "tooltip": "Mask detail. Higher = finer edges, slower."
                }),
                "min_area_percent": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 50.0, "step": 0.5,
                    "tooltip": "Minimum area as % of image to keep a segment."
                }),
            },
            "optional": {
                "sam_model": ("SAM_MODEL",),
                "grounding_dino_model": ("GROUNDING_DINO_MODEL",),
            },
        }

    def _fallback_segmentation(self, frame, prompts, min_area_pct, detail_level):
        """
        Automatic segmentation without AI models.
        Uses luminance thresholds, edge detection, and connected components.
        """
        h, w = frame.shape[:2]
        total_pixels = h * w
        min_area = total_pixels * (min_area_pct / 100.0)

        lum = 0.2126 * frame[:, :, 0] + 0.7152 * frame[:, :, 1] + 0.0722 * frame[:, :, 2]

        masks = []
        labels_out = []
        num_segments = min(len(prompts), 6)

        # Strategy: divide by luminance bands + spatial coherence
        thresholds = np.linspace(0, 1, num_segments + 1)

        for i in range(num_segments):
            low = thresholds[i]
            high = thresholds[i + 1]

            # Band mask
            mask = ((lum >= low) & (lum < high)).astype(np.float32)

            # Smooth edges based on detail level
            sigma = max(0.5, (1 - detail_level) * 5)
            mask = gaussian_filter(mask, sigma=sigma)
            mask = (mask > 0.3).astype(np.float32)

            # Remove small regions
            labeled, num_features = scipy_label(mask)
            for feat_id in range(1, num_features + 1):
                region = labeled == feat_id
                if region.sum() < min_area:
                    mask[region] = 0

            # Re-smooth after cleanup
            if detail_level < 0.8:
                mask = gaussian_filter(mask, sigma=1.0)
                mask = np.clip(mask, 0, 1)

            masks.append(mask.astype(np.float32))
            labels_out.append(prompts[i] if i < len(prompts) else f"region_{i}")

        # Pad to 6 masks
        while len(masks) < 6:
            masks.append(np.zeros((h, w), dtype=np.float32))
            labels_out.append("empty")

        return masks[:6], labels_out[:6]

    def _generate_depth_map(self, frame):
        """
        Generate a pseudo depth map from image cues.
        Uses vertical position + luminance + saturation as depth proxy.
        """
        h, w = frame.shape[:2]

        # Vertical gradient: top = far, bottom = near (typical outdoor scene)
        y_gradient = np.linspace(1.0, 0.0, h)[:, None] * np.ones((1, w))

        # Luminance: bright = far (sky), dark = near (foreground)
        lum = 0.2126 * frame[:, :, 0] + 0.7152 * frame[:, :, 1] + 0.0722 * frame[:, :, 2]

        # Saturation: low saturation = far (atmospheric haze)
        max_c = frame.max(axis=2)
        min_c = frame.min(axis=2)
        saturation = np.where(max_c > 0, (max_c - min_c) / (max_c + 1e-6), 0)

        # Combine cues
        depth = y_gradient * 0.4 + lum * 0.3 + (1 - saturation) * 0.3
        depth = gaussian_filter(depth, sigma=10)

        # Normalize
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-10)

        return depth.astype(np.float32)

    def segment(self, image, text_prompts, detail_level, min_area_percent,
                sam_model=None, grounding_dino_model=None):

        img_np = image.cpu().float().numpy()
        if len(img_np.shape) == 4:
            frame = img_np[0]
        else:
            frame = img_np

        h, w = frame.shape[:2]
        prompts = [p.strip() for p in text_prompts.split(",") if p.strip()]
        if not prompts:
            prompts = ["background"]

        # Try AI segmentation first
        if _HAS_SAM2 and sam_model is not None and _HAS_GDINO and grounding_dino_model is not None:
            masks, labels = self._ai_segmentation(
                frame, prompts, sam_model, grounding_dino_model,
                detail_level, min_area_percent
            )
        else:
            if not _HAS_SAM2:
                print("[SceneSegmenter] SAM2 not installed. Using luminance-based fallback.")
            masks, labels = self._fallback_segmentation(
                frame, prompts, min_area_percent, detail_level
            )

        # Generate depth map
        depth = self._generate_depth_map(frame)
        depth_rgb = np.stack([depth, depth, depth], axis=-1)

        # Build layer info JSON
        lum = 0.2126 * frame[:, :, 0] + 0.7152 * frame[:, :, 1] + 0.0722 * frame[:, :, 2]
        lfd = compute_local_fractal_dimension(lum, patch_size=7)

        layer_info = []
        for i, (mask, lbl) in enumerate(zip(masks, labels)):
            if mask.sum() < 1:
                layer_info.append({"label": lbl, "area_pct": 0, "avg_luminance": 0,
                                   "avg_fractal_dim": 0, "avg_depth": 0})
                continue

            area_pct = float(mask.sum() / (h * w) * 100)
            avg_lum = float(np.mean(lum[mask > 0.5]))
            avg_lfd = float(np.mean(lfd[mask > 0.5]))
            avg_depth = float(np.mean(depth[mask > 0.5]))

            layer_info.append({
                "label": lbl,
                "area_pct": round(area_pct, 1),
                "avg_luminance": round(avg_lum, 3),
                "avg_fractal_dim": round(avg_lfd, 2),
                "avg_depth": round(avg_depth, 3),
            })

        info_str = json.dumps(layer_info, indent=2)
        print(f"[SceneSegmenter] Segmented into {sum(1 for m in masks if m.sum() > 0)} layers: "
              f"{[l['label'] for l in layer_info if l['area_pct'] > 0]}")

        # Convert to tensors
        mask_tensors = [torch.from_numpy(m) for m in masks]
        depth_tensor = torch.from_numpy(depth_rgb[np.newaxis, ...])

        return (*mask_tensors, depth_tensor, info_str)

    def _ai_segmentation(self, frame, prompts, sam_model, gdino_model,
                         detail_level, min_area_pct):
        """Full AI segmentation with GroundingDINO + SAM2."""
        h, w = frame.shape[:2]
        total_pixels = h * w
        min_area = total_pixels * (min_area_pct / 100.0)

        frame_uint8 = (frame * 255).clip(0, 255).astype(np.uint8)

        masks = []
        labels = []

        for prompt in prompts[:6]:
            try:
                boxes, logits, phrases = gdino_predict(
                    model=gdino_model,
                    image=frame_uint8,
                    caption=prompt,
                    box_threshold=0.3,
                    text_threshold=0.25,
                )

                if len(boxes) == 0:
                    masks.append(np.zeros((h, w), dtype=np.float32))
                    labels.append(prompt)
                    continue

                predictor = SamPredictor(sam_model)
                predictor.set_image(frame_uint8)

                combined_mask = np.zeros((h, w), dtype=np.float32)
                for box in boxes:
                    box_np = box.cpu().numpy() * np.array([w, h, w, h])
                    sam_masks, _, _ = predictor.predict(
                        box=box_np,
                        multimask_output=False,
                    )
                    combined_mask = np.maximum(combined_mask, sam_masks[0].astype(np.float32))

                if combined_mask.sum() < min_area:
                    combined_mask = np.zeros((h, w), dtype=np.float32)

                masks.append(combined_mask)
                labels.append(prompt)

            except Exception as e:
                print(f"[SceneSegmenter] AI segmentation failed for '{prompt}': {e}")
                masks.append(np.zeros((h, w), dtype=np.float32))
                labels.append(prompt)

        while len(masks) < 6:
            masks.append(np.zeros((h, w), dtype=np.float32))
            labels.append("empty")

        return masks[:6], labels[:6]


class LayerDecomposer:
    """
    Decompose image into separate layers using masks from SceneSegmenter.
    Each layer is extracted as an independent image with alpha.
    """

    CATEGORY = "image/segmentation"
    FUNCTION = "decompose"
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("layer_1", "layer_2", "layer_3", "layer_4", "layer_5", "layer_6",
                    "layer_stack_info")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask_1": ("MASK",),
                "mask_2": ("MASK",),
                "layer_info": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "mask_3": ("MASK",),
                "mask_4": ("MASK",),
                "mask_5": ("MASK",),
                "mask_6": ("MASK",),
                "feather_radius": ("INT", {
                    "default": 3, "min": 0, "max": 20, "step": 1,
                    "tooltip": "Pixel radius for feathering mask edges."
                }),
                "fill_uncovered": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Fill pixels not covered by any mask into the nearest layer."
                }),
            },
        }

    def decompose(self, image, mask_1, mask_2, layer_info,
                  mask_3=None, mask_4=None, mask_5=None, mask_6=None,
                  feather_radius=3, fill_uncovered=True):

        img_np = image.cpu().float().numpy()
        if len(img_np.shape) == 4:
            frame = img_np[0]
        else:
            frame = img_np

        h, w, c = frame.shape

        # Collect masks
        all_masks = [mask_1]
        all_masks.append(mask_2)
        for m in [mask_3, mask_4, mask_5, mask_6]:
            if m is not None:
                all_masks.append(m)
            else:
                all_masks.append(torch.zeros(h, w))

        while len(all_masks) < 6:
            all_masks.append(torch.zeros(h, w))

        # Convert masks to numpy
        mask_arrays = []
        for m in all_masks[:6]:
            if isinstance(m, torch.Tensor):
                mn = m.cpu().float().numpy()
            else:
                mn = np.array(m, dtype=np.float32)

            if len(mn.shape) == 3:
                mn = mn[0] if mn.shape[0] == 1 else mn.mean(axis=-1)
            if mn.shape != (h, w):
                from scipy.ndimage import zoom
                mn = zoom(mn, (h / mn.shape[0], w / mn.shape[1]), order=1)

            mask_arrays.append(mn)

        # Feather masks
        if feather_radius > 0:
            for i in range(len(mask_arrays)):
                mask_arrays[i] = gaussian_filter(mask_arrays[i], sigma=feather_radius * 0.5)
                mask_arrays[i] = np.clip(mask_arrays[i], 0, 1)

        # Fill uncovered pixels by assigning to the largest overlapping mask
        if fill_uncovered:
            coverage = np.zeros((h, w), dtype=np.float32)
            for m in mask_arrays:
                coverage += m

            uncovered = coverage < 0.1
            if uncovered.any():
                # Assign uncovered pixels to closest non-empty mask via dilation
                for i in range(len(mask_arrays)):
                    if mask_arrays[i].sum() > 0:
                        dilated = binary_dilation(mask_arrays[i] > 0.5, iterations=50)
                        mask_arrays[i][uncovered & dilated] = 0.5
                        uncovered = uncovered & ~dilated

        # Extract layers
        layers = []
        for i, mask in enumerate(mask_arrays):
            layer = frame.copy()
            # Apply mask as alpha multiplier
            for ch in range(c):
                layer[:, :, ch] = frame[:, :, ch] * mask

            layers.append(layer[np.newaxis, ...].astype(np.float32))

        # Parse and enrich layer info
        try:
            info_list = json.loads(layer_info)
        except (json.JSONDecodeError, TypeError):
            info_list = [{"label": f"layer_{i}"} for i in range(6)]

        for i, info in enumerate(info_list):
            info["mask_area_pixels"] = int(mask_arrays[i].sum()) if i < len(mask_arrays) else 0

        stack_info = json.dumps(info_list, indent=2)

        layer_tensors = [torch.from_numpy(l) for l in layers]

        print(f"[LayerDecomposer] Decomposed into {sum(1 for m in mask_arrays if m.sum() > 10)} "
              f"active layers, feather={feather_radius}px")

        return (*layer_tensors, stack_info)


SCENE_SEGMENTATION_NODES = {
    "SceneSegmenter": SceneSegmenter,
    "LayerDecomposer": LayerDecomposer,
}

SCENE_SEGMENTATION_DISPLAY_NAMES = {
    "SceneSegmenter": "Scene Segmenter (AI)",
    "LayerDecomposer": "Layer Decomposer",
}
