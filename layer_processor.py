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


LAYER_PROCESSOR_NODES = {
    "LayerFractalProcessor": LayerFractalProcessor,
}

LAYER_PROCESSOR_DISPLAY_NAMES = {
    "LayerFractalProcessor": "Layer Fractal Processor",
}
