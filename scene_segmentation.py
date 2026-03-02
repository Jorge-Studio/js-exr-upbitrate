"""
AI Scene Segmentation nodes for ComfyUI (v4.1 - SAM 2.1 Tiered System).

4-tier detection backend:
  Tier 1 (sam3):       SAM 2.1 automatic mask generation via HuggingFace Transformers
  Tier 2 (dinox_sam3): DINO-X cloud API detection + SAM 2.1 local masks
  Tier 3 (gdino_sam3): Grounding DINO 1.0 local + SAM 2.1 local masks
  Tier 4 (fallback):   Luminance/edge-based (no AI needed)

SceneSegmenter outputs dynamic-length MASK lists (no fixed cap).
"""

import json
import os
import numpy as np
import torch
from PIL import Image as PILImage
from scipy.ndimage import gaussian_filter, label as scipy_label, binary_dilation

from .fractal_utils import compute_local_fractal_dimension

# ---------------------------------------------------------------------------
# Tier 1: SAM 3 via HuggingFace Transformers
# ---------------------------------------------------------------------------
_HAS_SAM3 = False
_sam3_model_cache = {}
_sam3_pipeline_cache = {}

try:
    from transformers import Sam2Processor, Sam2Model, pipeline as hf_pipeline
    _HAS_SAM3 = True
except ImportError:
    try:
        from transformers import Sam2Processor, Sam2Model
        hf_pipeline = None
        _HAS_SAM3 = True
    except ImportError:
        hf_pipeline = None

# ---------------------------------------------------------------------------
# Tier 2: DINO-X cloud API
# ---------------------------------------------------------------------------
_HAS_DINOX = False
try:
    from dds_cloudapi_sdk import Config as DINOXConfig, Client as DINOXClient
    from dds_cloudapi_sdk.tasks.dinox import DinoxTask
    _HAS_DINOX = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Tier 3: Legacy Grounding DINO 1.0
# ---------------------------------------------------------------------------
_HAS_GDINO = False
try:
    from groundingdino.util.inference import (
        load_model as load_gdino,
        predict as gdino_predict,
    )
    _HAS_GDINO = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Auto-describe vocabulary for broad scene scanning
# ---------------------------------------------------------------------------
AUTO_DESCRIBE_VOCAB = [
    "sky", "cloud", "sun", "moon",
    "tree", "trees", "foliage", "grass", "bush", "flower",
    "ground", "road", "path", "dirt", "sand", "snow",
    "water", "ocean", "river", "lake",
    "building", "house", "wall", "window", "roof",
    "person", "people", "face", "hand",
    "car", "vehicle", "truck", "train",
    "rock", "mountain", "cliff",
    "animal", "bird", "dog", "cat",
]


def _mask_to_numpy(mask_tensor, h, w):
    """Normalize any mask tensor/array to (h, w) float32 numpy."""
    if isinstance(mask_tensor, torch.Tensor):
        m = mask_tensor.cpu().float().numpy()
    else:
        m = np.array(mask_tensor, dtype=np.float32)
    if len(m.shape) == 3:
        m = m[0] if m.shape[0] == 1 else m.mean(axis=-1)
    if m.shape != (h, w):
        from scipy.ndimage import zoom as scipy_zoom
        m = scipy_zoom(m, (h / m.shape[0], w / m.shape[1]), order=1)
    return np.clip(m, 0, 1).astype(np.float32)


def _load_sam3(model_size="large"):
    """Load and cache SAM 2.1 model + processor."""
    if model_size in _sam3_model_cache:
        return _sam3_model_cache[model_size]

    size_map = {
        "large": "facebook/sam2.1-hiera-large",
        "base_plus": "facebook/sam2.1-hiera-base-plus",
        "tiny": "facebook/sam2.1-hiera-tiny",
    }
    model_id = size_map.get(model_size, "facebook/sam2.1-hiera-large")

    cache_dir = os.environ.get("HF_HOME", None)
    if cache_dir is None and os.path.isdir("/workspace/models/sam2"):
        cache_dir = "/workspace/models/sam2"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[SAM2] Loading {model_id} ({model_size}) on {device}...")

    kwargs = {"cache_dir": cache_dir} if cache_dir else {}
    model = Sam2Model.from_pretrained(model_id, **kwargs).to(device)
    processor = Sam2Processor.from_pretrained(model_id, **kwargs)
    model.eval()

    _sam3_model_cache[model_size] = (model, processor, device)
    print(f"[SAM2] Model loaded successfully.")
    return model, processor, device


def _classify_mask_heuristic(mask, frame, h, w):
    """Derive a semantic label from mask position, color, and area."""
    if mask.sum() < 10:
        return "unknown", 0.1

    ys, xs = np.where(mask > 0.5)
    if len(ys) == 0:
        return "unknown", 0.1

    cy = ys.mean() / h
    cx = xs.mean() / w
    area_pct = mask.sum() / (h * w)

    pixels = frame[mask > 0.5]
    avg_r = pixels[:, 0].mean() if len(pixels) > 0 else 0
    avg_g = pixels[:, 1].mean() if len(pixels) > 0 else 0
    avg_b = pixels[:, 2].mean() if len(pixels) > 0 else 0
    lum = 0.2126 * avg_r + 0.7152 * avg_g + 0.0722 * avg_b

    # Heuristic classification
    if cy < 0.35 and lum > 0.5 and area_pct > 0.1:
        return "sky", 0.7
    if cy < 0.3 and avg_b > avg_r and avg_b > avg_g:
        return "sky", 0.65
    if avg_g > avg_r * 1.2 and avg_g > avg_b * 1.2:
        if cy > 0.5:
            return "grass", 0.5
        return "foliage", 0.5
    if cy > 0.65 and lum < 0.4:
        return "ground", 0.5
    if cy > 0.7:
        return "ground", 0.4
    if area_pct < 0.05 and 0.3 < cy < 0.8:
        return "person", 0.3
    if lum > 0.7 and cy < 0.4:
        return "cloud", 0.4
    if avg_b > 0.4 and avg_b > avg_r and cy > 0.5:
        return "water", 0.35

    return "region", 0.2


# ============================================================================
# SceneSegmenter -- Main segmentation node with tiered backend
# ============================================================================
class SceneSegmenter:
    """
    Decompose an image into semantic masks using a 4-tier AI backend.

    Tier 1 (sam3): SAM 3 standalone -- best quality, native text prompts
    Tier 2 (dinox_sam3): DINO-X cloud detection + SAM 3 masks -- maximum accuracy
    Tier 3 (gdino_sam3): Grounding DINO 1.0 + SAM 3 masks -- fully offline
    Tier 4 (fallback): Luminance/edge segmentation -- no AI needed

    Outputs a dynamic-length MASK list (no fixed cap).
    """

    CATEGORY = "image/segmentation"
    FUNCTION = "segment"
    RETURN_TYPES = ("MASK", "IMAGE", "STRING", "INT")
    OUTPUT_IS_LIST = (True, False, False, False)
    RETURN_NAMES = ("masks", "depth_map", "layer_info", "layer_count")

    @classmethod
    def INPUT_TYPES(cls):
        backends = ["auto", "sam3", "dinox_sam3", "gdino_sam3", "fallback"]
        return {
            "required": {
                "image": ("IMAGE",),
                "text_prompts": ("STRING", {
                    "default": "sky, trees, ground, person, building",
                    "multiline": True,
                    "tooltip": "Comma-separated scene elements to segment. "
                               "Ignored when auto_describe is enabled."
                }),
                "detection_backend": (backends, {
                    "default": "auto",
                    "tooltip": "auto = best available. sam3 = SAM 3 standalone. "
                               "dinox_sam3 = DINO-X cloud + SAM 3. "
                               "gdino_sam3 = legacy GDINO + SAM 3. "
                               "fallback = luminance/edge."
                }),
                "model_size": (["large", "base_plus", "tiny"], {
                    "default": "large",
                    "tooltip": "SAM 3 model size. large = best, tiny = fastest."
                }),
                "auto_describe": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Auto-detect scene contents. "
                               "Overrides text_prompts with AI-detected labels."
                }),
                "detail_level": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1,
                    "tooltip": "Mask detail / confidence threshold."
                }),
                "min_area_percent": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 50.0, "step": 0.5,
                    "tooltip": "Minimum area as % of image to keep a segment."
                }),
            },
            "optional": {
                "dinox_api_key": ("STRING", {
                    "default": "",
                    "tooltip": "DINO-X API key for Tier 2 (get free at cloud.deepdataspace.com)."
                }),
            },
        }

    # ------------------------------------------------------------------
    # Tier selection
    # ------------------------------------------------------------------
    def _select_backend(self, requested, dinox_api_key):
        """Resolve 'auto' to the best available backend."""
        if requested != "auto":
            return requested

        if dinox_api_key and _HAS_DINOX and _HAS_SAM3:
            return "dinox_sam3"
        if _HAS_SAM3:
            return "sam3"
        if _HAS_GDINO and _HAS_SAM3:
            return "gdino_sam3"
        if _HAS_GDINO:
            return "gdino_sam3"
        return "fallback"

    # ------------------------------------------------------------------
    # Tier 1: SAM 2 automatic mask generation
    # ------------------------------------------------------------------
    def _sam3_segmentation(self, frame, prompts, model_size,
                           detail_level, min_area_pct, auto_describe):
        h, w = frame.shape[:2]
        total_pixels = h * w
        min_area = total_pixels * (min_area_pct / 100.0)

        pil_image = PILImage.fromarray((frame * 255).clip(0, 255).astype(np.uint8))

        masks = []
        labels = []
        scores_out = []

        size_map = {
            "large": "facebook/sam2.1-hiera-large",
            "base_plus": "facebook/sam2.1-hiera-base-plus",
            "tiny": "facebook/sam2.1-hiera-tiny",
        }
        model_id = size_map.get(model_size, "facebook/sam2.1-hiera-large")

        try:
            if hf_pipeline is not None:
                device_id = 0 if torch.cuda.is_available() else -1
                ppb = max(8, int(32 * detail_level))

                if model_id not in _sam3_pipeline_cache:
                    print(f"[SAM2] Creating mask-generation pipeline for {model_id}...")
                    _sam3_pipeline_cache[model_id] = hf_pipeline(
                        "mask-generation", model=model_id,
                        device=device_id, points_per_batch=ppb,
                    )
                generator = _sam3_pipeline_cache[model_id]
                result = generator(pil_image, points_per_batch=ppb)

                raw_masks = result.get("masks", [])
                raw_scores = result.get("scores", [])

                for i, m in enumerate(raw_masks):
                    mask_np = np.array(m, dtype=np.float32)
                    if mask_np.shape != (h, w):
                        from scipy.ndimage import zoom as scipy_zoom
                        mask_np = scipy_zoom(mask_np, (h / mask_np.shape[0], w / mask_np.shape[1]), order=1)
                    mask_np = (mask_np > 0.5).astype(np.float32)

                    if mask_np.sum() < min_area:
                        continue

                    score = float(raw_scores[i]) if i < len(raw_scores) else 0.5
                    label, conf = _classify_mask_heuristic(mask_np, frame, h, w)
                    score = max(score, conf)

                    masks.append(mask_np)
                    labels.append(label)
                    scores_out.append(score)
            else:
                model, processor, device = _load_sam3(model_size)
                grid_size = max(4, int(8 * detail_level))
                points = []
                for gy in range(grid_size):
                    for gx in range(grid_size):
                        px = int((gx + 0.5) / grid_size * w)
                        py = int((gy + 0.5) / grid_size * h)
                        points.append([px, py])

                for pt in points:
                    try:
                        input_points = [[[[pt[0], pt[1]]]]]
                        input_labels = [[[1]]]
                        inputs = processor(
                            images=pil_image,
                            input_points=input_points,
                            input_labels=input_labels,
                            return_tensors="pt",
                        ).to(device)

                        with torch.no_grad():
                            outputs = model(**inputs)

                        pred_masks = processor.post_process_masks(
                            outputs.pred_masks.cpu(),
                            inputs["original_sizes"],
                        )

                        if len(pred_masks) > 0 and pred_masks[0].shape[0] > 0:
                            best_idx = outputs.iou_scores.squeeze().argmax().item()
                            m = pred_masks[0][0, best_idx].float().numpy()
                            if m.shape != (h, w):
                                from scipy.ndimage import zoom as scipy_zoom
                                m = scipy_zoom(m, (h / m.shape[0], w / m.shape[1]), order=1)
                            m = (m > 0.5).astype(np.float32)

                            if m.sum() >= min_area:
                                is_duplicate = False
                                for existing in masks:
                                    iou = (m * existing).sum() / max((m + existing - m * existing).sum(), 1)
                                    if iou > 0.8:
                                        is_duplicate = True
                                        break
                                if not is_duplicate:
                                    label, conf = _classify_mask_heuristic(m, frame, h, w)
                                    masks.append(m)
                                    labels.append(label)
                                    scores_out.append(conf)
                    except Exception:
                        continue

        except Exception as e:
            import traceback
            print(f"[SceneSegmenter] SAM 2 auto-mask error: {type(e).__name__}: {e}")
            traceback.print_exc()
            return self._fallback_segmentation(
                frame, prompts, min_area_pct, detail_level, auto_describe
            )

        if len(masks) == 0:
            return self._fallback_segmentation(
                frame, prompts, min_area_pct, detail_level, auto_describe
            )

        return masks, labels, scores_out

    # ------------------------------------------------------------------
    # Tier 2: DINO-X cloud + SAM 2 refinement
    # ------------------------------------------------------------------
    def _dinox_sam3_segmentation(self, frame, prompts, api_key, model_size,
                                 detail_level, min_area_pct, auto_describe):
        h, w = frame.shape[:2]
        total_pixels = h * w
        min_area = total_pixels * (min_area_pct / 100.0)

        import tempfile
        pil_image = PILImage.fromarray((frame * 255).clip(0, 255).astype(np.uint8))

        masks = []
        labels = []
        scores_out = []

        try:
            config = DINOXConfig(token=api_key)
            client = DINOXClient(config)

            tmp_path = os.path.join(tempfile.gettempdir(), "_dinox_tmp.jpg")
            pil_image.save(tmp_path, quality=95)

            if auto_describe:
                prompt_text = ". ".join(AUTO_DESCRIBE_VOCAB[:20])
            else:
                prompt_text = ". ".join(prompts)

            task = DinoxTask(
                image_url=tmp_path,
                prompts=[{"type": "text", "text": prompt_text}],
            )
            client.run_task(task)
            result = task.result

            if not result or not hasattr(result, "objects"):
                print("[SceneSegmenter] DINO-X returned no objects, falling back to SAM 2")
                return self._sam3_segmentation(
                    frame, prompts, model_size, detail_level, min_area_pct, auto_describe
                )

            model, processor, device = _load_sam3(model_size)

            for obj in result.objects:
                box = obj.bbox
                label = getattr(obj, "category", "") or getattr(obj, "caption", "object")
                score = getattr(obj, "score", 0.5)

                if score < 0.2:
                    continue

                input_boxes = [[list(box)]]

                inputs = processor(
                    images=pil_image,
                    input_boxes=input_boxes,
                    return_tensors="pt",
                ).to(device)

                with torch.no_grad():
                    outputs = model(**inputs)

                pred_masks = processor.post_process_masks(
                    outputs.pred_masks.cpu(),
                    inputs["original_sizes"],
                )

                if len(pred_masks) > 0 and pred_masks[0].shape[0] > 0:
                    best_idx = outputs.iou_scores.squeeze().argmax().item() if outputs.iou_scores.numel() > 1 else 0
                    best_mask = pred_masks[0][0, best_idx].float().numpy()
                    if best_mask.shape != (h, w):
                        from scipy.ndimage import zoom as scipy_zoom
                        best_mask = scipy_zoom(best_mask, (h / best_mask.shape[0], w / best_mask.shape[1]), order=1)
                    best_mask = (best_mask > 0.5).astype(np.float32)

                    if best_mask.sum() >= min_area:
                        masks.append(best_mask)
                        labels.append(label)
                        scores_out.append(float(score))

            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        except Exception as e:
            print(f"[SceneSegmenter] DINO-X error: {e}, falling back to SAM 2")
            return self._sam3_segmentation(
                frame, prompts, model_size, detail_level, min_area_pct, auto_describe
            )

        return masks, labels, scores_out

    # ------------------------------------------------------------------
    # Tier 3: Legacy Grounding DINO 1.0 + SAM 2 masks
    # ------------------------------------------------------------------
    def _gdino_sam3_segmentation(self, frame, prompts, model_size,
                                  detail_level, min_area_pct, auto_describe):
        h, w = frame.shape[:2]
        total_pixels = h * w
        min_area = total_pixels * (min_area_pct / 100.0)

        frame_uint8 = (frame * 255).clip(0, 255).astype(np.uint8)
        pil_image = PILImage.fromarray(frame_uint8)

        if auto_describe:
            prompts = AUTO_DESCRIBE_VOCAB[:20]

        if _HAS_SAM3:
            model, processor, device = _load_sam3(model_size)
        else:
            model, processor, device = None, None, "cpu"

        masks = []
        labels = []
        scores_out = []

        for prompt in prompts:
            try:
                boxes, logits, phrases = gdino_predict(
                    model=None,
                    image=frame_uint8,
                    caption=prompt,
                    box_threshold=0.3,
                    text_threshold=0.25,
                )

                if len(boxes) == 0:
                    if not auto_describe:
                        masks.append(np.zeros((h, w), dtype=np.float32))
                        labels.append(prompt)
                        scores_out.append(0.0)
                    continue

                combined_mask = np.zeros((h, w), dtype=np.float32)

                if _HAS_SAM3:
                    for box in boxes:
                        box_np = box.cpu().numpy() * np.array([w, h, w, h])
                        input_boxes = [[box_np.tolist()]]

                        inputs = processor(
                            images=pil_image,
                            input_boxes=input_boxes,
                            return_tensors="pt",
                        ).to(device)

                        with torch.no_grad():
                            outputs = model(**inputs)

                        pred_masks = processor.post_process_masks(
                            outputs.pred_masks.cpu(),
                            inputs["original_sizes"],
                        )

                        if len(pred_masks) > 0 and pred_masks[0].shape[0] > 0:
                            best_idx = outputs.iou_scores.squeeze().argmax().item() if outputs.iou_scores.numel() > 1 else 0
                            m = pred_masks[0][0, best_idx].float().numpy()
                            if m.shape != (h, w):
                                from scipy.ndimage import zoom as scipy_zoom
                                m = scipy_zoom(m, (h / m.shape[0], w / m.shape[1]), order=1)
                            combined_mask = np.maximum(combined_mask, (m > 0.5).astype(np.float32))

                if combined_mask.sum() < min_area:
                    if not auto_describe:
                        combined_mask = np.zeros((h, w), dtype=np.float32)
                    else:
                        continue

                score = float(logits.max()) if len(logits) > 0 else 0.5
                masks.append(combined_mask)
                labels.append(prompt)
                scores_out.append(score)

            except Exception as e:
                print(f"[SceneSegmenter] GDINO+SAM2 error for '{prompt}': {e}")
                if not auto_describe:
                    masks.append(np.zeros((h, w), dtype=np.float32))
                    labels.append(prompt)
                    scores_out.append(0.0)

        return masks, labels, scores_out

    # ------------------------------------------------------------------
    # Tier 4: Luminance/edge fallback (no AI)
    # ------------------------------------------------------------------
    def _fallback_segmentation(self, frame, prompts, min_area_pct,
                               detail_level, auto_describe):
        h, w = frame.shape[:2]
        total_pixels = h * w
        min_area = total_pixels * (min_area_pct / 100.0)

        lum = 0.2126 * frame[:, :, 0] + 0.7152 * frame[:, :, 1] + 0.0722 * frame[:, :, 2]

        num_segments = max(3, min(len(prompts), 12)) if not auto_describe else 6
        thresholds = np.linspace(0, 1, num_segments + 1)

        masks = []
        labels_out = []
        scores_out = []

        for i in range(num_segments):
            low = thresholds[i]
            high = thresholds[i + 1]

            mask = ((lum >= low) & (lum < high)).astype(np.float32)

            sigma = max(0.5, (1 - detail_level) * 5)
            mask = gaussian_filter(mask, sigma=sigma)
            mask = (mask > 0.3).astype(np.float32)

            labeled, num_features = scipy_label(mask)
            for feat_id in range(1, num_features + 1):
                region = labeled == feat_id
                if region.sum() < min_area:
                    mask[region] = 0

            if detail_level < 0.8:
                mask = gaussian_filter(mask, sigma=1.0)
                mask = np.clip(mask, 0, 1)

            if mask.sum() < min_area:
                continue

            if auto_describe:
                label, conf = _classify_mask_heuristic(mask, frame, h, w)
            else:
                label = prompts[i] if i < len(prompts) else f"region_{i}"
                conf = 0.3

            masks.append(mask.astype(np.float32))
            labels_out.append(label)
            scores_out.append(conf)

        return masks, labels_out, scores_out

    # ------------------------------------------------------------------
    # Depth map
    # ------------------------------------------------------------------
    def _generate_depth_map(self, frame):
        h, w = frame.shape[:2]

        y_gradient = np.linspace(1.0, 0.0, h)[:, None] * np.ones((1, w))
        lum = 0.2126 * frame[:, :, 0] + 0.7152 * frame[:, :, 1] + 0.0722 * frame[:, :, 2]
        max_c = frame.max(axis=2)
        min_c = frame.min(axis=2)
        saturation = np.where(max_c > 0, (max_c - min_c) / (max_c + 1e-6), 0)

        depth = y_gradient * 0.4 + lum * 0.3 + (1 - saturation) * 0.3
        depth = gaussian_filter(depth, sigma=10)
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-10)

        return depth.astype(np.float32)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def segment(self, image, text_prompts, detection_backend, model_size,
                auto_describe, detail_level, min_area_percent, dinox_api_key=""):

        img_np = image.cpu().float().numpy()
        if len(img_np.shape) == 4:
            frame = img_np[0]
        else:
            frame = img_np

        h, w = frame.shape[:2]
        prompts = [p.strip() for p in text_prompts.split(",") if p.strip()]
        if not prompts and not auto_describe:
            prompts = ["background"]

        backend = self._select_backend(detection_backend, dinox_api_key)
        print(f"[SceneSegmenter] Using backend: {backend}")

        if backend == "sam3" and _HAS_SAM3:
            masks, labels, scores = self._sam3_segmentation(
                frame, prompts, model_size, detail_level, min_area_percent, auto_describe
            )
        elif backend == "dinox_sam3" and _HAS_DINOX and _HAS_SAM3:
            masks, labels, scores = self._dinox_sam3_segmentation(
                frame, prompts, dinox_api_key, model_size,
                detail_level, min_area_percent, auto_describe
            )
        elif backend == "gdino_sam3" and _HAS_GDINO:
            masks, labels, scores = self._gdino_sam3_segmentation(
                frame, prompts, model_size, detail_level, min_area_percent, auto_describe
            )
        else:
            if backend not in ("fallback",):
                print(f"[SceneSegmenter] Backend '{backend}' unavailable, using fallback.")
            masks, labels, scores = self._fallback_segmentation(
                frame, prompts, min_area_percent, detail_level, auto_describe
            )

        # Ensure at least one mask
        if len(masks) == 0:
            masks = [np.ones((h, w), dtype=np.float32)]
            labels = ["background"]
            scores = [1.0]

        # Generate depth map
        depth = self._generate_depth_map(frame)
        depth_rgb = np.stack([depth, depth, depth], axis=-1)

        # Build layer info JSON
        lum = 0.2126 * frame[:, :, 0] + 0.7152 * frame[:, :, 1] + 0.0722 * frame[:, :, 2]
        lfd = compute_local_fractal_dimension(lum, patch_size=7)

        from .layer_processor import _match_label_to_preset

        layer_info = []
        for i, (mask, lbl) in enumerate(zip(masks, labels)):
            if mask.sum() < 1:
                layer_info.append({
                    "label": lbl, "area_pct": 0, "confidence": 0,
                    "avg_luminance": 0, "avg_fractal_dim": 0, "avg_depth": 0,
                    "fractal_preset": "default",
                    "detail_prompt": f"detailed {lbl}, cinematic, high quality",
                })
                continue

            area_pct = float(mask.sum() / (h * w) * 100)
            avg_lum = float(np.mean(lum[mask > 0.5]))
            avg_lfd = float(np.mean(lfd[mask > 0.5]))
            avg_depth = float(np.mean(depth[mask > 0.5]))
            conf = scores[i] if i < len(scores) else 0.5
            preset = _match_label_to_preset(lbl)

            layer_info.append({
                "label": lbl,
                "area_pct": round(area_pct, 1),
                "confidence": round(float(conf), 2),
                "avg_luminance": round(avg_lum, 3),
                "avg_fractal_dim": round(avg_lfd, 2),
                "avg_depth": round(avg_depth, 3),
                "fractal_preset": preset,
                "detail_prompt": f"detailed {lbl}, cinematic, high quality",
            })

        info_str = json.dumps(layer_info, indent=2)
        active = [l["label"] for l in layer_info if l["area_pct"] > 0]
        print(f"[SceneSegmenter] {backend}: {len(masks)} masks, "
              f"active: {active}")

        # Convert to tensors -- return as list for OUTPUT_IS_LIST
        mask_tensors = [torch.from_numpy(m) for m in masks]
        depth_tensor = torch.from_numpy(depth_rgb[np.newaxis, ...])

        return (mask_tensors, depth_tensor, info_str, len(masks))


# ============================================================================
# LayerSelector -- pick one layer from a dynamic list
# ============================================================================
class LayerSelector:
    """Extract a single MASK from a dynamic mask list by index."""

    CATEGORY = "image/segmentation"
    FUNCTION = "select"
    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("mask", "layer_info_entry")
    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),
                "layer_info": ("STRING",),
                "index": ("INT", {"default": 0, "min": 0, "max": 99}),
            },
        }

    def select(self, masks, layer_info, index):
        idx = index[0] if isinstance(index, list) else index
        info_str = layer_info[0] if isinstance(layer_info, list) else layer_info

        idx = int(idx)
        if idx >= len(masks):
            print(f"[LayerSelector] Index {idx} out of range ({len(masks)} masks), clamping")
            idx = len(masks) - 1
        if idx < 0:
            idx = 0

        mask = masks[idx]

        try:
            info_list = json.loads(info_str)
            entry = info_list[idx] if idx < len(info_list) else {"label": f"layer_{idx}"}
        except (json.JSONDecodeError, TypeError):
            entry = {"label": f"layer_{idx}"}

        return (mask, json.dumps(entry))


# ============================================================================
# LayerDecomposer -- extract layers from mask list
# ============================================================================
class LayerDecomposer:
    """
    Decompose image into separate layers using a dynamic mask list.
    Each layer is the original image multiplied by its mask.
    """

    CATEGORY = "image/segmentation"
    FUNCTION = "decompose"
    RETURN_TYPES = ("IMAGE", "STRING")
    OUTPUT_IS_LIST = (True, False)
    RETURN_NAMES = ("layers", "layer_stack_info")
    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "masks": ("MASK",),
                "layer_info": ("STRING",),
            },
            "optional": {
                "feather_radius": ("INT", {
                    "default": 3, "min": 0, "max": 20, "step": 1,
                }),
            },
        }

    def decompose(self, image, masks, layer_info, feather_radius=None):
        img_list = image if isinstance(image, list) else [image]
        img_t = img_list[0]
        img_np = img_t.cpu().float().numpy()
        if len(img_np.shape) == 4:
            frame = img_np[0]
        else:
            frame = img_np
        h, w, c = frame.shape

        info_str = layer_info[0] if isinstance(layer_info, list) else layer_info
        fr = 3
        if feather_radius is not None:
            fr = feather_radius[0] if isinstance(feather_radius, list) else feather_radius

        mask_list = masks if isinstance(masks, list) else [masks]

        # Convert and feather masks
        mask_arrays = []
        for m_t in mask_list:
            m = _mask_to_numpy(m_t, h, w)
            if fr > 0:
                m = gaussian_filter(m, sigma=fr * 0.5)
                m = np.clip(m, 0, 1)
            mask_arrays.append(m)

        # Extract layers
        layers = []
        for mask in mask_arrays:
            layer = frame.copy()
            for ch in range(c):
                layer[:, :, ch] = frame[:, :, ch] * mask
            layers.append(torch.from_numpy(layer[np.newaxis, ...].astype(np.float32)))

        try:
            info_list = json.loads(info_str)
        except (json.JSONDecodeError, TypeError):
            info_list = [{"label": f"layer_{i}"} for i in range(len(mask_arrays))]

        for i, info in enumerate(info_list):
            if i < len(mask_arrays):
                info["mask_area_pixels"] = int(mask_arrays[i].sum())

        stack_info = json.dumps(info_list, indent=2)

        active = sum(1 for m in mask_arrays if m.sum() > 10)
        print(f"[LayerDecomposer] Decomposed into {active} active layers, feather={fr}px")

        return (layers, stack_info)


# ============================================================================
# SegmentationPreview -- visualize all layers with color overlay
# ============================================================================
class SegmentationPreview:
    """Visualize all segmented layers with color-coded overlay and labels."""

    CATEGORY = "image/segmentation"
    FUNCTION = "preview"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("preview",)
    INPUT_IS_LIST = True

    LAYER_COLORS = [
        (66, 133, 244), (52, 168, 83), (183, 129, 56),
        (234, 67, 53), (154, 103, 215), (251, 188, 4),
        (0, 188, 212), (255, 112, 67), (121, 134, 203),
        (255, 213, 79), (77, 182, 172), (239, 83, 80),
        (171, 71, 188), (255, 167, 38), (102, 187, 106),
        (66, 165, 245), (141, 110, 99), (189, 189, 189),
        (255, 138, 101), (100, 181, 246),
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "masks": ("MASK",),
                "layer_info": ("STRING",),
                "overlay_opacity": ("FLOAT", {
                    "default": 0.45, "min": 0.1, "max": 0.9, "step": 0.05,
                }),
                "show_labels": ("BOOLEAN", {"default": True}),
            },
        }

    def preview(self, image, masks, layer_info, overlay_opacity, show_labels):
        from PIL import ImageDraw, ImageFont

        img_list = image if isinstance(image, list) else [image]
        img_np = img_list[0].cpu().float().numpy()
        if len(img_np.shape) == 4:
            frame = img_np[0]
        else:
            frame = img_np
        h, w, c = frame.shape

        info_str = layer_info[0] if isinstance(layer_info, list) else layer_info
        opacity = overlay_opacity[0] if isinstance(overlay_opacity, list) else overlay_opacity
        do_labels = show_labels[0] if isinstance(show_labels, list) else show_labels

        mask_list = masks if isinstance(masks, list) else [masks]

        try:
            info_list = json.loads(info_str)
        except (json.JSONDecodeError, TypeError):
            info_list = [{"label": f"layer_{i}"} for i in range(len(mask_list))]

        overlay = np.zeros((h, w, 3), dtype=np.float64)
        total_mask = np.zeros((h, w), dtype=np.float64)

        mask_arrays = []
        for mask_t in mask_list:
            m = _mask_to_numpy(mask_t, h, w)
            mask_arrays.append(m)

        for i, m in enumerate(mask_arrays):
            if m.sum() < 1:
                continue
            color = self.LAYER_COLORS[i % len(self.LAYER_COLORS)]
            for ch_idx in range(3):
                overlay[:, :, ch_idx] += m * (color[ch_idx] / 255.0)
            total_mask += m

        total_mask = np.clip(total_mask, 0, 1)

        result = frame.copy().astype(np.float64)
        for ch_idx in range(3):
            result[:, :, ch_idx] = (
                frame[:, :, ch_idx] * (1 - opacity * total_mask) +
                overlay[:, :, ch_idx] * opacity
            )
        result = np.clip(result, 0, 1)

        if do_labels:
            pil_img = PILImage.fromarray((result * 255).astype(np.uint8))
            draw = ImageDraw.Draw(pil_img)

            font = None
            for fp in [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/System/Library/Fonts/Helvetica.ttc",
                "C:/Windows/Fonts/arial.ttf",
            ]:
                try:
                    font = ImageFont.truetype(fp, max(16, h // 30))
                    break
                except (OSError, IOError):
                    continue
            if font is None:
                font = ImageFont.load_default()

            for i, m in enumerate(mask_arrays):
                if m.sum() < 10:
                    continue
                ys, xs = np.where(m > 0.5)
                if len(ys) == 0:
                    continue
                cy, cx = int(ys.mean()), int(xs.mean())

                entry = info_list[i] if i < len(info_list) else {}
                label = entry.get("label", f"layer_{i}")
                area = entry.get("area_pct", 0)
                conf = entry.get("confidence", 0)
                lfd_val = entry.get("avg_fractal_dim", 0)

                text = label
                if area > 0:
                    text += f" ({area:.0f}%)"
                if conf > 0:
                    text += f" [{conf:.0%}]"
                if lfd_val > 0:
                    text += f" LFD:{lfd_val:.1f}"

                color = self.LAYER_COLORS[i % len(self.LAYER_COLORS)]
                bbox = draw.textbbox((cx, cy), text, font=font)
                pad = 4
                draw.rectangle(
                    [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad],
                    fill=(0, 0, 0, 180),
                )
                draw.text((cx, cy), text, fill=color, font=font)

            result = np.array(pil_img).astype(np.float32) / 255.0

        output = result[np.newaxis, ...].astype(np.float32)
        print(f"[SegmentationPreview] Rendered {len(mask_arrays)} layers")
        return (torch.from_numpy(output),)


# ============================================================================
# LayerInpaintPrepare -- unchanged single-layer node
# ============================================================================
class LayerInpaintPrepare:
    """Prepare a segmented layer for text-to-image inpainting via KSampler."""

    CATEGORY = "image/segmentation"
    FUNCTION = "prepare"
    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("inpaint_image", "inpaint_mask", "suggested_prompt")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "layer_mask": ("MASK",),
                "layer_label": ("STRING", {"default": "sky"}),
                "expand_mask_px": ("INT", {
                    "default": 10, "min": 0, "max": 50, "step": 2,
                }),
                "detail_prompt_style": (
                    ["photorealistic", "cinematic", "natural", "custom"],
                    {"default": "cinematic"},
                ),
            },
            "optional": {
                "custom_prompt": ("STRING", {"default": "", "multiline": True}),
            },
        }

    PROMPT_TEMPLATES = {
        "sky": {
            "photorealistic": "detailed blue sky with subtle cloud wisps, photorealistic, 8k",
            "cinematic": "cinematic sky, atmospheric perspective, film grain, anamorphic",
            "natural": "natural sky gradient, soft light",
        },
        "cloud": {
            "photorealistic": "detailed cumulus clouds, volumetric lighting, 8k",
            "cinematic": "dramatic cloud formations, golden hour light, cinematic",
            "natural": "soft natural clouds",
        },
        "trees": {
            "photorealistic": "detailed tree foliage, individual leaves visible, photorealistic 8k",
            "cinematic": "cinematic forest canopy, depth of field, atmospheric haze",
            "natural": "natural tree detail, organic textures",
        },
        "foliage": {
            "photorealistic": "lush vegetation detail, leaf textures, photorealistic",
            "cinematic": "cinematic vegetation, volumetric light through leaves",
            "natural": "natural plant detail, organic",
        },
        "ground": {
            "photorealistic": "detailed ground texture, dirt and pebbles, 8k macro",
            "cinematic": "cinematic ground plane, shallow depth of field",
            "natural": "natural ground detail, earth textures",
        },
        "person": {
            "photorealistic": "detailed human features, skin texture, fabric detail, 8k",
            "cinematic": "cinematic portrait lighting, film emulation, anamorphic bokeh",
            "natural": "natural skin tones, fabric texture",
        },
        "building": {
            "photorealistic": "architectural detail, surface textures, brick and mortar, 8k",
            "cinematic": "cinematic architecture, dramatic lighting, production design",
            "natural": "natural building textures, weathered surfaces",
        },
        "water": {
            "photorealistic": "detailed water surface, caustics, reflections, 8k",
            "cinematic": "cinematic water, light play on surface, anamorphic",
            "natural": "natural water ripples, soft reflections",
        },
        "skin": {
            "photorealistic": "detailed skin texture, pores visible, subsurface scattering, 8k",
            "cinematic": "cinematic skin tones, beauty lighting, film emulation",
            "natural": "natural skin detail, soft focus",
        },
        "face": {
            "photorealistic": "detailed facial features, skin pores, catch light in eyes, 8k",
            "cinematic": "cinematic close-up, dramatic lighting, shallow DOF",
            "natural": "natural face detail, soft light",
        },
    }

    def _get_prompt(self, label, style, custom_prompt):
        if style == "custom" and custom_prompt:
            return custom_prompt
        label_lower = label.lower().strip()
        if label_lower in self.PROMPT_TEMPLATES:
            return self.PROMPT_TEMPLATES[label_lower].get(
                style, f"detailed {label}, high quality"
            )
        for key, templates in self.PROMPT_TEMPLATES.items():
            if key in label_lower or label_lower in key:
                return templates.get(style, f"detailed {label}")
        return f"detailed {label}, high quality, {style}"

    def prepare(self, image, layer_mask, layer_label, expand_mask_px,
                detail_prompt_style, custom_prompt=""):
        img_np = image.cpu().float().numpy()
        if len(img_np.shape) == 3:
            img_np = img_np[np.newaxis, ...]

        mask_np = layer_mask.cpu().float().numpy()
        if len(mask_np.shape) == 3:
            mask_np = mask_np[0] if mask_np.shape[0] == 1 else mask_np.mean(axis=-1)

        h, w = img_np.shape[1], img_np.shape[2]
        if mask_np.shape != (h, w):
            from scipy.ndimage import zoom as scipy_zoom
            mask_np = scipy_zoom(mask_np, (h / mask_np.shape[0], w / mask_np.shape[1]), order=1)

        if expand_mask_px > 0:
            struct = np.ones((expand_mask_px * 2 + 1, expand_mask_px * 2 + 1))
            expanded = binary_dilation(mask_np > 0.5, structure=struct).astype(np.float32)
            expanded = gaussian_filter(expanded, sigma=expand_mask_px * 0.3)
            mask_np = np.clip(expanded, 0, 1)

        prompt = self._get_prompt(layer_label, detail_prompt_style, custom_prompt)
        mask_tensor = torch.from_numpy(mask_np[np.newaxis, ...])

        print(f"[LayerInpaintPrepare] '{layer_label}' -> {detail_prompt_style} prompt")
        return (image, mask_tensor, prompt)


# ============================================================================
# Node registrations
# ============================================================================
SCENE_SEGMENTATION_NODES = {
    "SceneSegmenter": SceneSegmenter,
    "LayerSelector": LayerSelector,
    "LayerDecomposer": LayerDecomposer,
    "SegmentationPreview": SegmentationPreview,
    "LayerInpaintPrepare": LayerInpaintPrepare,
}

SCENE_SEGMENTATION_DISPLAY_NAMES = {
    "SceneSegmenter": "Scene Segmenter (SAM3 Tiered)",
    "LayerSelector": "Layer Selector",
    "LayerDecomposer": "Layer Decomposer",
    "SegmentationPreview": "Segmentation Preview",
    "LayerInpaintPrepare": "Layer Inpaint Prepare",
}
