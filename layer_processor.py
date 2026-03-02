"""
Per-Layer Fractal Processing for ComfyUI.

LayerFractalProcessor: Applies fractal bit-depth expansion with parameters
automatically tuned for the semantic content of each layer (sky, foliage,
skin, structures, etc.).
"""

import json
import numpy as np
import torch

from .fractal_utils import (
    compute_local_fractal_dimension,
    fractal_brownian_motion,
    hermite_interpolate_neighbors,
    sobel_gradient,
    temporal_coherent_seed,
    generate_blue_noise,
)

# Optimal fractal parameters per semantic category.
# Tuned for professional colourist expectations.
LAYER_PRESETS = {
    "sky": {
        "fractal_octaves": 2,
        "fractal_persistence": 0.3,
        "gradient_smoothness": 0.95,
        "description": "Ultra-smooth gradient fill. Minimal texture."
    },
    "cloud": {
        "fractal_octaves": 3,
        "fractal_persistence": 0.45,
        "gradient_smoothness": 0.75,
        "description": "Soft organic edges with subtle detail."
    },
    "water": {
        "fractal_octaves": 4,
        "fractal_persistence": 0.45,
        "gradient_smoothness": 0.8,
        "description": "Smooth base with subtle ripple texture."
    },
    "skin": {
        "fractal_octaves": 3,
        "fractal_persistence": 0.4,
        "gradient_smoothness": 0.7,
        "description": "Subsurface-aware fill. Preserves skin tone gradients."
    },
    "face": {
        "fractal_octaves": 3,
        "fractal_persistence": 0.35,
        "gradient_smoothness": 0.75,
        "description": "Extra smooth for facial tones."
    },
    "hair": {
        "fractal_octaves": 6,
        "fractal_persistence": 0.55,
        "gradient_smoothness": 0.25,
        "description": "High-detail fractal for strand-level texture."
    },
    "foliage": {
        "fractal_octaves": 6,
        "fractal_persistence": 0.6,
        "gradient_smoothness": 0.3,
        "description": "Rich organic micro-texture for leaves/vegetation."
    },
    "trees": {
        "fractal_octaves": 6,
        "fractal_persistence": 0.6,
        "gradient_smoothness": 0.3,
        "description": "Same as foliage - complex natural texture."
    },
    "grass": {
        "fractal_octaves": 5,
        "fractal_persistence": 0.55,
        "gradient_smoothness": 0.35,
        "description": "Dense organic texture."
    },
    "ground": {
        "fractal_octaves": 5,
        "fractal_persistence": 0.5,
        "gradient_smoothness": 0.4,
        "description": "Medium texture for earth/dirt/sand."
    },
    "rock": {
        "fractal_octaves": 5,
        "fractal_persistence": 0.55,
        "gradient_smoothness": 0.35,
        "description": "Rough geological texture."
    },
    "concrete": {
        "fractal_octaves": 4,
        "fractal_persistence": 0.45,
        "gradient_smoothness": 0.5,
        "description": "Flat with subtle surface noise."
    },
    "building": {
        "fractal_octaves": 3,
        "fractal_persistence": 0.35,
        "gradient_smoothness": 0.6,
        "description": "Sharp edges, smooth surfaces."
    },
    "metal": {
        "fractal_octaves": 3,
        "fractal_persistence": 0.3,
        "gradient_smoothness": 0.65,
        "description": "Smooth reflective surfaces."
    },
    "fabric": {
        "fractal_octaves": 5,
        "fractal_persistence": 0.5,
        "gradient_smoothness": 0.4,
        "description": "Woven micro-texture."
    },
    "person": {
        "fractal_octaves": 4,
        "fractal_persistence": 0.45,
        "gradient_smoothness": 0.55,
        "description": "Mixed skin/clothing. Balanced approach."
    },
    "default": {
        "fractal_octaves": 4,
        "fractal_persistence": 0.5,
        "gradient_smoothness": 0.5,
        "description": "Balanced default for unknown content."
    },
}

PRESET_NAMES = list(LAYER_PRESETS.keys())


def _match_label_to_preset(label):
    """Find the closest preset for a given label string."""
    label_lower = label.lower().strip()

    # Exact match
    if label_lower in LAYER_PRESETS:
        return label_lower

    # Partial match
    for preset_name in LAYER_PRESETS:
        if preset_name in label_lower or label_lower in preset_name:
            return preset_name

    # Keyword heuristics
    keywords = {
        "sky": ["sky", "heaven", "atmosphere", "blue"],
        "cloud": ["cloud", "cumulus", "overcast"],
        "water": ["water", "ocean", "sea", "lake", "river", "pool"],
        "skin": ["skin", "body", "hand", "arm", "leg"],
        "face": ["face", "head", "portrait"],
        "hair": ["hair", "fur", "beard"],
        "foliage": ["leaf", "leaves", "bush", "vegetation", "plant", "flower"],
        "trees": ["tree", "forest", "wood", "branch", "trunk"],
        "grass": ["grass", "lawn", "field", "meadow"],
        "ground": ["ground", "earth", "dirt", "sand", "floor", "path"],
        "rock": ["rock", "stone", "cliff", "mountain", "boulder"],
        "building": ["building", "house", "wall", "window", "door", "roof", "structure"],
        "metal": ["metal", "steel", "iron", "car", "vehicle"],
        "fabric": ["fabric", "cloth", "clothing", "dress", "shirt"],
    }

    for preset, kws in keywords.items():
        for kw in kws:
            if kw in label_lower:
                return preset

    return "default"


class LayerFractalProcessor:
    """
    Apply fractal bit-depth expansion to a single layer with parameters
    auto-tuned based on the layer's semantic label.
    """

    CATEGORY = "image/bitdepth"
    FUNCTION = "process_layer"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("expanded_layer", "preset_used")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "layer_image": ("IMAGE",),
                "layer_mask": ("MASK",),
                "layer_label": ("STRING", {
                    "default": "default",
                    "tooltip": "Semantic label (sky, foliage, skin, etc.). "
                               "Used for auto-parameter selection."
                }),
                "auto_params": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Auto-detect optimal fractal parameters from label."
                }),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**31 - 1}),
            },
            "optional": {
                "fractal_octaves": ("INT", {
                    "default": 4, "min": 1, "max": 8, "step": 1
                }),
                "fractal_persistence": ("FLOAT", {
                    "default": 0.5, "min": 0.1, "max": 0.9, "step": 0.05
                }),
                "gradient_smoothness": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05
                }),
            },
        }

    def process_layer(self, layer_image, layer_mask, layer_label, auto_params, seed,
                      fractal_octaves=4, fractal_persistence=0.5, gradient_smoothness=0.5):

        img_np = layer_image.cpu().float().numpy()
        if len(img_np.shape) == 3:
            img_np = img_np[np.newaxis, ...]

        mask_np = layer_mask.cpu().float().numpy()
        if len(mask_np.shape) == 3:
            mask_np = mask_np[0] if mask_np.shape[0] == 1 else mask_np.mean(axis=-1)

        # Determine parameters
        if auto_params:
            preset_key = _match_label_to_preset(layer_label)
            preset = LAYER_PRESETS[preset_key]
            octaves = preset["fractal_octaves"]
            persistence = preset["fractal_persistence"]
            smoothness = preset["gradient_smoothness"]
            desc = preset["description"]
        else:
            preset_key = "manual"
            octaves = fractal_octaves
            persistence = fractal_persistence
            smoothness = gradient_smoothness
            desc = "Manual parameters"

        print(f"[LayerFractalProcessor] '{layer_label}' -> preset: {preset_key} "
              f"(octaves={octaves}, persist={persistence}, smooth={smoothness})")

        results = []
        for frame_idx in range(img_np.shape[0]):
            frame = img_np[frame_idx]
            h, w, c = frame.shape

            # Resize mask if needed
            m = mask_np
            if m.shape != (h, w):
                from scipy.ndimage import zoom
                m = zoom(m, (h / m.shape[0], w / m.shape[1]), order=1)

            # Compute LFD for this layer
            lum = 0.2126 * frame[:, :, 0] + 0.7152 * frame[:, :, 1] + 0.0722 * frame[:, :, 2]
            lfd_map = compute_local_fractal_dimension(lum, patch_size=7)

            # Generate fractal noise
            frame_seed = temporal_coherent_seed(seed, frame_idx, 0.8)
            fbm = fractal_brownian_motion((h, w), octaves=octaves,
                                          persistence=persistence,
                                          seed=frame_seed)

            step = 1.0 / 255.0
            expanded = frame.astype(np.float64)

            for ch in range(c):
                channel = frame[:, :, ch].astype(np.float64)

                # Gradient-based offset
                grad_x, grad_y = sobel_gradient(channel)
                grad_dir = (grad_x + grad_y)
                gmax = np.abs(grad_dir).max()
                if gmax > 0:
                    grad_dir = grad_dir / gmax
                gradient_offset = grad_dir * step * 0.35

                # Fractal micro-texture
                fractal_scale = np.clip(lfd_map / 2.0, 0, 1)
                fractal_offset = fbm * step * 0.3 * fractal_scale

                # Smooth Hermite offset
                smooth_off = hermite_interpolate_neighbors(frame, channel=ch) * step * 0.25

                # Blend by smoothness parameter
                fractal_weight = 1 - smoothness
                smooth_weight = smoothness
                offset = smooth_off * smooth_weight + fractal_offset * fractal_weight + gradient_offset

                # Apply only within mask
                expanded[:, :, ch] = channel + offset * m

            expanded = np.clip(expanded, 0.0, 1.0).astype(np.float32)
            results.append(expanded)

        output = np.stack(results, axis=0)
        preset_info = json.dumps({
            "label": layer_label,
            "preset": preset_key,
            "octaves": octaves,
            "persistence": persistence,
            "smoothness": smoothness,
            "description": desc,
        })

        return (torch.from_numpy(output), preset_info)


class LayerDetailEditor:
    """
    Per-layer control for editing AI inference results before fractal processing.

    Lets the user override the auto-detected label, scale fractal/smooth
    strength, change the detail prompt, or skip the layer entirely.
    Designed to sit between LayerSelector and LayerFractalProcessor.
    """

    CATEGORY = "image/segmentation"
    FUNCTION = "edit"
    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("layer_image", "layer_mask", "layer_info_entry")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "layer_image": ("IMAGE",),
                "layer_mask": ("MASK",),
                "layer_info_entry": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "override_label": ("STRING", {
                    "default": "",
                    "tooltip": "Override auto-detected label. Blank = keep auto."
                }),
                "fractal_strength_mult": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 3.0, "step": 0.1,
                    "tooltip": "Multiplier for fractal octaves/persistence."
                }),
                "smooth_strength_mult": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 3.0, "step": 0.1,
                    "tooltip": "Multiplier for gradient smoothness."
                }),
                "detail_prompt_override": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "Custom detail prompt for inpainting. Blank = auto."
                }),
                "skip_layer": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Zero out mask so downstream ignores this layer."
                }),
            },
        }

    def edit(self, layer_image, layer_mask, layer_info_entry,
             override_label="", fractal_strength_mult=1.0,
             smooth_strength_mult=1.0, detail_prompt_override="",
             skip_layer=False):

        try:
            entry = json.loads(layer_info_entry)
        except (json.JSONDecodeError, TypeError):
            entry = {"label": "unknown"}

        if skip_layer:
            h = layer_image.shape[-3] if len(layer_image.shape) == 4 else layer_image.shape[0]
            w = layer_image.shape[-2] if len(layer_image.shape) == 4 else layer_image.shape[1]
            zero_mask = torch.zeros_like(layer_mask)
            entry["skipped"] = True
            print(f"[LayerDetailEditor] Skipping layer '{entry.get('label', '?')}'")
            return (layer_image, zero_mask, json.dumps(entry))

        if override_label.strip():
            old_label = entry.get("label", "")
            entry["label"] = override_label.strip()
            entry["original_label"] = old_label
            new_preset = _match_label_to_preset(override_label.strip())
            entry["fractal_preset"] = new_preset
            print(f"[LayerDetailEditor] Label override: '{old_label}' -> '{override_label.strip()}' (preset: {new_preset})")

        if fractal_strength_mult != 1.0:
            entry["fractal_strength_mult"] = round(fractal_strength_mult, 2)

        if smooth_strength_mult != 1.0:
            entry["smooth_strength_mult"] = round(smooth_strength_mult, 2)

        if detail_prompt_override.strip():
            entry["detail_prompt"] = detail_prompt_override.strip()

        return (layer_image, layer_mask, json.dumps(entry))


class BatchLayerFractalProcessor:
    """
    One-click path that processes ALL layers from SceneSegmenter at once.

    Iterates over the dynamic layer list, looks up fractal presets per label,
    and applies fractal bit-depth expansion. Accepts and returns lists.

    Pipeline: SceneSegmenter -> BatchLayerFractalProcessor -> LayerAssembler
    """

    CATEGORY = "image/bitdepth"
    FUNCTION = "process_all"
    RETURN_TYPES = ("IMAGE",)
    OUTPUT_IS_LIST = (True,)
    RETURN_NAMES = ("processed_layers",)
    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "masks": ("MASK",),
                "layer_info": ("STRING",),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**31 - 1}),
            },
            "optional": {
                "global_fractal_mult": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1,
                    "tooltip": "Global multiplier for fractal strength across all layers."
                }),
                "global_smooth_mult": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1,
                    "tooltip": "Global multiplier for gradient smoothness across all layers."
                }),
            },
        }

    def process_all(self, images, masks, layer_info, seed,
                    global_fractal_mult=None, global_smooth_mult=None):

        image_list = images if isinstance(images, list) else [images]
        mask_list = masks if isinstance(masks, list) else [masks]
        info_str = layer_info[0] if isinstance(layer_info, list) else layer_info
        seed_val = seed[0] if isinstance(seed, list) else seed

        gfm = 1.0
        if global_fractal_mult is not None:
            gfm = global_fractal_mult[0] if isinstance(global_fractal_mult, list) else global_fractal_mult
        gsm = 1.0
        if global_smooth_mult is not None:
            gsm = global_smooth_mult[0] if isinstance(global_smooth_mult, list) else global_smooth_mult

        try:
            info_list = json.loads(info_str)
        except (json.JSONDecodeError, TypeError):
            info_list = [{"label": f"layer_{i}"} for i in range(len(image_list))]

        # We need the original full image for Hermite interpolation reference.
        # Use the first image in the list as the reference frame.
        ref_img_np = image_list[0].cpu().float().numpy()
        if len(ref_img_np.shape) == 4:
            ref_frame = ref_img_np[0]
        else:
            ref_frame = ref_img_np

        results = []
        num_layers = min(len(image_list), len(mask_list))

        for i in range(num_layers):
            entry = info_list[i] if i < len(info_list) else {"label": f"layer_{i}"}
            label = entry.get("label", f"layer_{i}")

            if entry.get("skipped", False):
                results.append(image_list[i])
                continue

            preset_key = entry.get("fractal_preset", _match_label_to_preset(label))
            preset = LAYER_PRESETS.get(preset_key, LAYER_PRESETS["default"])

            f_mult = entry.get("fractal_strength_mult", 1.0) * gfm
            s_mult = entry.get("smooth_strength_mult", 1.0) * gsm

            octaves = min(8, max(1, int(preset["fractal_octaves"] * f_mult)))
            persistence = np.clip(preset["fractal_persistence"] * f_mult, 0.1, 0.9)
            smoothness = np.clip(preset["gradient_smoothness"] * s_mult, 0.0, 1.0)

            img_np = image_list[i].cpu().float().numpy()
            if len(img_np.shape) == 3:
                img_np = img_np[np.newaxis, ...]

            mask_np = mask_list[i].cpu().float().numpy() if isinstance(mask_list[i], torch.Tensor) else np.array(mask_list[i])
            if len(mask_np.shape) == 3:
                mask_np = mask_np[0] if mask_np.shape[0] == 1 else mask_np.mean(axis=-1)

            frame = img_np[0]
            h, w, c = frame.shape

            if mask_np.shape != (h, w):
                from scipy.ndimage import zoom
                mask_np = zoom(mask_np, (h / mask_np.shape[0], w / mask_np.shape[1]), order=1)

            lum = 0.2126 * frame[:, :, 0] + 0.7152 * frame[:, :, 1] + 0.0722 * frame[:, :, 2]
            lfd_map = compute_local_fractal_dimension(lum, patch_size=7)

            frame_seed = temporal_coherent_seed(int(seed_val), i, 0.8)
            fbm = fractal_brownian_motion((h, w), octaves=octaves,
                                          persistence=persistence,
                                          seed=frame_seed)

            step = 1.0 / 255.0
            expanded = frame.astype(np.float64)

            for ch in range(c):
                channel = frame[:, :, ch].astype(np.float64)

                grad_x, grad_y = sobel_gradient(channel)
                grad_dir = grad_x + grad_y
                gmax = np.abs(grad_dir).max()
                if gmax > 0:
                    grad_dir = grad_dir / gmax
                gradient_offset = grad_dir * step * 0.35

                fractal_scale = np.clip(lfd_map / 2.0, 0, 1)
                fractal_offset = fbm * step * 0.3 * fractal_scale

                smooth_off = hermite_interpolate_neighbors(ref_frame, channel=ch) * step * 0.25

                fractal_weight = 1 - smoothness
                smooth_weight = smoothness
                offset = smooth_off * smooth_weight + fractal_offset * fractal_weight + gradient_offset

                expanded[:, :, ch] = channel + offset * mask_np

            expanded = np.clip(expanded, 0.0, 1.0).astype(np.float32)
            results.append(torch.from_numpy(expanded[np.newaxis, ...]))

            print(f"[BatchLayerFractalProcessor] Layer {i} '{label}' -> "
                  f"preset={preset_key}, octaves={octaves}, "
                  f"persist={persistence:.2f}, smooth={smoothness:.2f}")

        print(f"[BatchLayerFractalProcessor] Processed {len(results)} layers total")
        return (results,)


LAYER_PROCESSOR_NODES = {
    "LayerFractalProcessor": LayerFractalProcessor,
    "LayerDetailEditor": LayerDetailEditor,
    "BatchLayerFractalProcessor": BatchLayerFractalProcessor,
}

LAYER_PROCESSOR_DISPLAY_NAMES = {
    "LayerFractalProcessor": "Layer Fractal Processor",
    "LayerDetailEditor": "Layer Detail Editor",
    "BatchLayerFractalProcessor": "Batch Layer Fractal Processor",
}
