# High Bit Depth EXR Export & Cinema Delivery for ComfyUI (v4.1)

Professional-grade EXR export with **SAM 3 Tiered AI Segmentation**, **Fractal Bit-Depth Expansion**, **Per-Layer Editing**, **Log format support**, **color grading controls**, **cinema delivery compliance**, **motion animation**, **luminance-preserving deflicker**, and **maximum tonal precision** for VFX, film, compositing, and color grading workflows.

**v4.1: SAM 3 Tiered Segmentation + Auto-Describe + Dynamic Layers + Per-Layer Edit**

---

## Quick Start

1. Install the node (see Installation below)
2. Load any included workflow from `workflows/`
3. Connect your image and run!

**Recommended workflows:**
- `SAM3_Tiered_AutoSegment.json` - **NEW** One-click auto-segmentation + batch fractal processing
- `SAM3_PerLayer_Edit.json` - **NEW** Per-layer controls with LayerDetailEditor
- `Fractal_BitDepth_Full.json` - Updated full pipeline using SAM 3 tiered backend
- `Fractal_BitDepth_NoAI.json` - Fractal-only 8-to-32-bit expansion (no AI models needed)
- `Segmentation_Preview_Inpaint.json` - Segmentation preview + text-to-image inpainting setup
- `ultimate_exr_workflow.json` - Complete HDR processing pipeline

---

## What's New in v4.1

### SAM 3 Tiered Segmentation System (NEW!)

Replaced the SAM v1 + GroundingDINO backend with a 4-tier system:

| Tier | Backend | Quality | Requires | Use Case |
|------|---------|---------|----------|----------|
| **1: sam3** | SAM 3 standalone (HuggingFace) | Excellent -- native text prompts, 2x better than SAM2 | `transformers>=4.40.0`, GPU, ~2.5GB | Default for most users |
| **2: dinox_sam3** | DINO-X cloud + SAM 3 local masks | Best possible -- 59.8 AP LVIS detection | DINO-X API key, internet | Maximum detection accuracy |
| **3: gdino_sam3** | Grounding DINO 1.0 + SAM 3 masks | Very good -- proven local detector | `groundingdino` package | Fully offline, legacy |
| **4: fallback** | Luminance/edge thresholds | Basic | No AI, CPU only | Environments without GPU |

**Auto selection**: Set `detection_backend` to `"auto"` and the node will use the best available tier.

### Auto-Describe Mode (NEW!)

Enable `auto_describe` on SceneSegmenter to automatically detect and label scene elements without manual text prompts. Uses spatial heuristics (y-position, color, area) and AI confidence scores. Outputs editable `layer_info` JSON with auto-detected labels, confidence scores, fractal presets, and detail prompts.

### Dynamic Layer Count (NEW!)

Removed the fixed 6-mask output cap. SceneSegmenter now outputs a dynamic-length MASK list (via `OUTPUT_IS_LIST`). Downstream nodes accept lists via `INPUT_IS_LIST`. The number of layers adapts to the scene automatically.

### New Nodes

| Node | Description |
|------|-------------|
| **Layer Selector** | Extract a single mask from the dynamic list by index |
| **Layer Detail Editor** | Override labels, scale fractal/smooth strength, skip layers |
| **Batch Layer Fractal Processor** | One-click processing of ALL layers at once |

### Updated Nodes

| Node | Change |
|------|--------|
| **Scene Segmenter** | Tiered backend, auto-describe, dynamic mask list output |
| **Layer Decomposer** | Accepts mask lists via `INPUT_IS_LIST` |
| **Segmentation Preview** | Accepts mask lists, 20 color palette, confidence display |
| **Layer Assembler** | Accepts layer+mask lists, no more fixed 6-pair limit |

---

## DINO-X API Setup (Tier 2)

For maximum detection accuracy, set up DINO-X cloud:

1. Get a free API key at https://cloud.deepdataspace.com
2. Install the SDK: `pip install dds-cloudapi-sdk`
3. In the SceneSegmenter node, set `detection_backend` to `"dinox_sam3"` and paste your key in `dinox_api_key`

DINO-X achieves 59.8 AP on LVIS -- the best open-set detection available. It handles the detection, then SAM 3 runs locally for pixel-precise masks.

---

## What's in v4.0

### Fractal Bit-Depth Expansion

Expand 8-bit source material to genuine 32-bit float using fractal mathematics:

- **Fractal Bit-Depth Expander**: Uses Local Fractal Dimension (LFD) analysis, fractal Brownian motion (fBm), Hermite spline interpolation, and rational fractal cubic splines.
- **Perceptual Dither**: Blue noise, TPDF, or fractal dither patterns for banding-free gradients.

### Shadow-Controlled HDR

- **Shadow Controlled Exposure**: Power-curve shadow control with noise suppression.
- **Shadow Curve Processor**: Non-linear curves that peak instead of roll off.
- **Rec.709 Converter**: Accurate Rec.709 conversion using `colour-science`.

### Exposure Bracketing

- **Exposure Bracket Generator**: Generate multiple EV levels from a single source.
- **Exposure Bracket to TIFF**: Save 16-bit TIFF brackets.
- **Video to Exposure Brackets**: Process video into bracket sequences.

---

## What's in v3.x

### Luminance-Preserving Deflicker
- **Luminance Deflicker**: Fix brightness flicker **without blur** (gain_only, histogram_match)
- **Normals Deflicker**: Stabilize gradients while preserving luminance

### Cinema Delivery (Molinare Compliant)
- **Save EXR Sequence**: ACES 2065-1, DCI 4K, numbered sequence
- **Generate Delivery CSV**: Professional manifest files
- **ACES to Rec.709 Preview**: View-transform for monitoring

### Animation & Motion Control
- **Animated Pan & Scan**: Keyframe-based pan, zoom, rotation
- **Load EXR Image / Sequence**: Load HDR EXR files
- **Extract / Apply Motion Path**: Optical flow motion extraction

### Core Color & EXR
- **Log Format Export**: ARRI LogC3, Sony S-Log3, V-Log, Canon Log 3, RED Log3G10, DaVinci Intermediate
- **Color Space Converter**, **Color Grading Controller**, **HDR Curve Editor**
- **Color Match to Reference**, **Advanced Color Match** (5 algorithms), **Auto Exposure Match**
- **Image Stats**: Verify range, unique values, effective bit depth

---

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Jorge-Studio/js-exr-upbitrate.git
cd js-exr-upbitrate
pip install -r requirements.txt
# Restart ComfyUI
```

### Dependencies

| Package | Required | Purpose |
|---------|----------|---------|
| openexr | **Yes** | True 32-bit EXR writing |
| imath | **Yes** | OpenEXR dependency |
| pyexr | **Yes** | EXR reading/writing (primary) |
| numpy | **Yes** | Core math / array ops |
| scipy | **Yes** | Gaussian filter, LFD, morphology |
| pillow | **Yes** | Image handling, font rendering |
| imageio | Recommended | EXR fallback + video handling |
| opencv-python | Recommended | Debanding, rotation, optical flow, TIFF |
| colour-science | Recommended | Accurate Rec.709/ACES color transforms |
| **transformers** | **Recommended** | **SAM 3 model (Tier 1 segmentation)** |
| torch | **Yes** | Tensor operations (bundled with ComfyUI) |

#### Optional AI Backends

| Backend | Package | Purpose |
|---------|---------|---------|
| SAM 3 (Tier 1) | `transformers>=4.40.0` | Primary segmentation (auto-downloads ~2.5GB model) |
| DINO-X (Tier 2) | `dds-cloudapi-sdk` | Cloud detection API -- best accuracy, needs API key |
| Grounding DINO (Tier 3) | `groundingdino` | Legacy local detector -- fully offline |

The Scene Segmenter falls back to luminance/edge-based segmentation (Tier 4) if no AI models are installed.

---

## All Nodes (38 Total)

### Core Color & EXR Nodes (10)

| Node | Description |
|------|-------------|
| **Prepare Image High Bit Depth** | sRGB -> Linear + headroom + debanding |
| **Color Grading Controller** | Exposure, contrast, lift/gamma/gain, saturation |
| **HDR Curve Editor** | Shadows, midtones, highlights, whites, blacks |
| **Color Match to Reference** | Auto-match processed image colors to original |
| **Advanced Color Match** | 5 algorithms: Histogram, LAB, Reinhard, CLAHE, CDF |
| **Auto Exposure Match** | Quick exposure-only matching |
| **Color Space Converter** | Convert between sRGB, Linear, and Log formats |
| **Save Image EXR** | Export single image as 16/32-bit EXR with Log format |
| **Save Video EXR Sequence** | Export video as EXR sequence with Log format |
| **Image Stats** | Display range, unique values, bit depth estimate |

### Fractal Bit-Depth Expansion Nodes (2)

| Node | Description |
|------|-------------|
| **Fractal Bit-Depth Expander** | 8-to-32-bit expansion via LFD, fBm, Hermite splines |
| **Perceptual Dither** | Blue noise, TPDF, or fractal dither |

### AI Scene Segmentation Nodes (5)

| Node | Description |
|------|-------------|
| **Scene Segmenter (SAM3 Tiered)** | 4-tier backend, auto-describe, dynamic mask list output |
| **Layer Selector** | **NEW** Extract single mask from dynamic list by index |
| **Layer Decomposer** | Extract layers with alpha masks, per-layer stats |
| **Segmentation Preview** | Color-coded overlay with labels, confidence, LFD |
| **Layer Inpaint Prepare** | Per-segment image + mask + prompt for KSampler |

### Layer Processing Nodes (5)

| Node | Description |
|------|-------------|
| **Layer Fractal Processor** | Per-layer fractal expansion with semantic auto-tuning |
| **Layer Detail Editor** | **NEW** Override labels, strength multipliers, skip layers |
| **Batch Layer Fractal Processor** | **NEW** One-click all-layers processing |
| **Layer Detail Enhancer** | Laplacian pyramid frequency blending |
| **Layer Assembler** | Composite layers with alpha feathering + ACES conversion |

### Validation (1)

| Node | Description |
|------|-------------|
| **Bit-Depth Validator** | QC: unique values, PSNR, SSIM, gradient smoothness, waveform |

### Shadow-Controlled HDR Nodes (3)

| Node | Description |
|------|-------------|
| **Shadow Controlled Exposure** | Power-curve shadow control |
| **Shadow Curve Processor** | Non-linear peaking curves |
| **Rec.709 Converter** | Accurate Rec.709 via colour-science |

### Exposure Bracketing Nodes (3)

| Node | Description |
|------|-------------|
| **Exposure Bracket Generator** | Multiple EV brackets from single source |
| **Exposure Bracket to TIFF** | Save 16-bit TIFF brackets |
| **Video to Exposure Brackets** | Video bracket sequences |

### Cinema Delivery Nodes (3)

| Node | Description |
|------|-------------|
| **Save EXR Sequence (Cinema)** | DCI 4K ACES 2065-1 EXR delivery |
| **Generate Delivery CSV** | Professional manifest |
| **ACES to Rec.709 Preview** | RRT+ODT view transform |

### Animation & Motion Nodes (5)

| Node | Description |
|------|-------------|
| **Animated Pan & Scan** | Keyframe animation with easing |
| **Load EXR Image** | Load single HDR EXR files |
| **Load EXR Sequence** | Load EXR sequence as video |
| **Extract Motion from Video** | Optical flow motion extraction |
| **Apply Motion Path** | Apply motion to sequences |

### Deflicker Nodes (2)

| Node | Description |
|------|-------------|
| **Luminance Deflicker** | Per-frame gain correction (NO BLUR) |
| **Normals Deflicker** | Gradient-preserving temporal smoothing |

---

## Pipeline Examples

### One-Click Auto-Segmentation Pipeline

```
[Source Image]
    |
Scene Segmenter (auto_describe=true, backend=auto)
    |
    +--> Segmentation Preview --> PreviewImage
    |
    +--> BatchLayerFractalProcessor (processes ALL layers at once)
            |
            LayerAssembler (composite back)
                |
                SaveImageEXR (32-bit, ACES)
```

### Per-Layer Edit Pipeline

```
[Source Image]
    |
Scene Segmenter (text_prompts: "sky, trees, ground")
    |
    +--> LayerSelector (index=0) --> LayerDetailEditor (sky: low fractal)
    |                                    |
    |                                    LayerFractalProcessor
    |
    +--> LayerSelector (index=1) --> LayerDetailEditor (trees: high fractal)
    |                                    |
    |                                    LayerFractalProcessor
    |
    +--> LayerSelector (index=2) --> LayerDetailEditor (ground: medium)
                                         |
                                         LayerFractalProcessor
    |
    LayerAssembler --> BitDepthValidator --> SaveImageEXR
```

### Semantic-Aware Auto-Tuning (Layer Fractal Processor)

| Layer Type | Fractal Octaves | Persistence | Smoothness | Strategy |
|-----------|----------------|-------------|------------|----------|
| Sky | 2 | 0.30 | 0.95 | Ultra-smooth gradient fill |
| Cloud | 3 | 0.45 | 0.75 | Soft organic edges |
| Skin / Face | 3 | 0.35-0.40 | 0.70-0.75 | Subsurface-aware, no artifacts |
| Foliage / Trees | 6 | 0.60 | 0.30 | Rich organic micro-texture |
| Ground | 5 | 0.50 | 0.40 | Medium earth texture |
| Building | 3 | 0.35 | 0.60 | Sharp edges, smooth surfaces |
| Water | 4 | 0.45 | 0.80 | Smooth with ripple texture |
| Hair | 6 | 0.55 | 0.25 | Strand-level detail |

---

## Node Settings Reference

### Scene Segmenter (SAM3 Tiered)

| Setting | Default | Description |
|---------|---------|-------------|
| text_prompts | "sky, trees, ground, person, building" | Comma-separated labels (ignored when auto_describe=true) |
| detection_backend | "auto" | auto / sam3 / dinox_sam3 / gdino_sam3 / fallback |
| model_size | "large" | SAM 3 model size: large, base_plus, tiny |
| auto_describe | false | Auto-detect scene contents |
| detail_level | 0.5 | Mask detail / confidence threshold (0-1) |
| min_area_percent | 1.0 | Minimum segment area as % of image |
| dinox_api_key | "" | Optional DINO-X API key for Tier 2 |

### Layer Selector

| Setting | Default | Description |
|---------|---------|-------------|
| index | 0 | Which mask to extract from the list (0-based) |

### Layer Detail Editor

| Setting | Default | Description |
|---------|---------|-------------|
| override_label | "" | Replace auto-detected label (blank = keep) |
| fractal_strength_mult | 1.0 | Multiplier for fractal parameters (0-3) |
| smooth_strength_mult | 1.0 | Multiplier for smoothness (0-3) |
| detail_prompt_override | "" | Custom inpainting prompt (blank = auto) |
| skip_layer | false | Zero out mask so downstream ignores this layer |

### Batch Layer Fractal Processor

| Setting | Default | Description |
|---------|---------|-------------|
| seed | 42 | Random seed for fractal noise |
| global_fractal_mult | 1.0 | Scale fractal strength across ALL layers |
| global_smooth_mult | 1.0 | Scale smoothness across ALL layers |

### Layer Assembler

| Setting | Default | Description |
|---------|---------|-------------|
| feather_radius | 3 | Mask feathering for seamless compositing |
| output_colorspace | passthrough | Convert output: passthrough, sRGB, Rec.709, Linear, ACEScg, ACES2065-1 |
| background_color | 0.0 | Fill color for uncovered areas |

### Segmentation Preview

| Setting | Default | Description |
|---------|---------|-------------|
| overlay_opacity | 0.45 | Color overlay strength (0.1-0.9) |
| show_labels | true | Draw labels with area%, confidence, LFD |

---

## Molinare/Professional DI Delivery

### Supported Specifications
- **Resolution**: DCI 4K (4096x2160), UHD 4K, 2K, 1080p
- **Color Space**: ACES 2065-1, ACEScct, Linear Rec.709
- **Bit Depth**: 16-bit half-float or 32-bit full float
- **Compression**: PIZ (recommended), ZIP, ZIPS, RLE, None
- **Naming**: `shot_name_V###.####.exr`

### Cinema Delivery Workflow

```
[Video/Image Source]
    |
Prepare Image High Bit Depth
    |
Color Grading Controller
    |
Save EXR Sequence (Cinema)
    |-- shot_name: "KSA_001_010"
    |-- convert_to_aces: true
    |-- bit_depth: 16 / compression: piz
    |
Generate Delivery CSV
```

---

## Included Workflows (27)

### SAM3 Segmentation Workflows (NEW!)
| Workflow | Description |
|----------|-------------|
| `SAM3_Tiered_AutoSegment.json` | One-click auto-segment + batch fractal processing |
| `SAM3_PerLayer_Edit.json` | Per-layer editing with LayerDetailEditor |

### Fractal & Segmentation Workflows
| Workflow | Description |
|----------|-------------|
| `Fractal_BitDepth_Full.json` | Updated full pipeline using SAM 3 tiered backend |
| `Fractal_BitDepth_NoAI.json` | Fractal-only 8-to-32-bit expansion |
| `Segmentation_Preview_Inpaint.json` | Preview + text-to-image inpaint setup |

### Bit-Depth & HDR Workflows
| Workflow | Description |
|----------|-------------|
| `BitDepth_From_Single_Source.json` | Exposure bracketing + HDR merge |
| `Anti_Flicker_HDR_Pipeline.json` | Anti-flicker HDR processing |

### Core Workflows
| Workflow | Description |
|----------|-------------|
| `ultimate_exr_workflow.json` | Complete HDR pipeline |
| `advanced_color_match_workflow.json` | Compare all 5 color match algorithms |

### Cinema Delivery
| Workflow | Description |
|----------|-------------|
| `Test1-6, Molinare, Video_*` | Cinema delivery, video export, and test workflows |

### Deflicker
| Workflow | Description |
|----------|-------------|
| `Luminance_Deflicker_NoBlur.json` | Brightness correction without blur |
| `Normals_Deflicker_GradPreserve.json` | Gradient-based flicker reduction |
| `Quick_Deflicker_Video.json` | Quick video deflicker pipeline |

---

## File Structure

```
js-exr-upbitrate/
  __init__.py                 # Core nodes + registration (38 nodes)
  fractal_utils.py            # Fractal math library (LFD, fBm, Hermite, blue noise)
  fractal_bitdepth.py         # FractalBitDepthExpander, PerceptualDither
  scene_segmentation.py       # SceneSegmenter (SAM3 tiered), LayerSelector,
                              # LayerDecomposer, SegmentationPreview, LayerInpaintPrepare
  layer_processor.py          # LayerFractalProcessor, LayerDetailEditor,
                              # BatchLayerFractalProcessor
  ai_detail_layer.py          # LayerDetailEnhancer
  layer_assembly.py           # LayerAssembler, BitDepthValidator
  cinema_delivery.py          # SaveEXRSequence, GenerateDeliveryCSV, ACESToRec709Preview
  animated_motion.py          # AnimatedPanAndScan, LoadEXR*, MotionPath*
  luminance_deflicker.py      # LuminanceDeflicker, NormalsDeflicker
  exposure_bracketing.py      # ExposureBracketGenerator, ExposureBracketToTIFF, VideoToExposureBrackets
  shadow_controlled_hdr.py    # ShadowControlledExposure, ShadowCurveProcessor, Rec709Converter
  requirements.txt            # Python dependencies
  workflows/                  # 27 ready-to-use workflow JSONs
  js/                         # ComfyUI web extensions
```

---

## Troubleshooting

### Scene Segmenter shows "fallback" only
AI models not installed. Install `transformers>=4.40.0` for SAM 3 (Tier 1).

### DINO-X returns errors
- Verify your API key at https://cloud.deepdataspace.com
- Check internet connection -- DINO-X is a cloud API
- The node will automatically fall back to SAM 3 standalone

### SAM 3 model download is slow
First run downloads ~2.5GB from HuggingFace. For RunPod, download to the network volume for persistence:
```bash
pip install -U transformers
python -c "from transformers import Sam3Model; Sam3Model.from_pretrained('facebook/sam3')"
```

### "Cannot load EXR" error
Install pyexr: `pip install pyexr`

### Color looks wrong in Resolve
- Verify Input Color Space matches export format
- For ACES 2065-1 output, set Input to "ACES 2065-1 (AP0)"

### colour-science API errors
Update to latest: `pip install --upgrade colour-science`

---

## License

MIT

---

## Links

- **GitHub**: https://github.com/Jorge-Studio/js-exr-upbitrate
- **ComfyUI**: https://github.com/comfyanonymous/ComfyUI
- **Molinare**: https://www.molinare.co.uk/

---

## Credits

Developed by **Jorge Studio / KS Films** for professional AI-to-cinema pipelines.

Part of the **NodyJS** ecosystem for ComfyUI.
