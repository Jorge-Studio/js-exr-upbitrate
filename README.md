# JS EXR Upbitrate — Cinema-Grade Bit-Depth & AI Segmentation for ComfyUI (v4.2)

Professional-grade EXR export with **SAM 2.1 Tiered AI Segmentation**, **Fractal Bit-Depth Expansion**, **Per-Layer Editing**, **Log format support**, **color grading controls**, **cinema delivery compliance**, **motion animation**, **luminance-preserving deflicker**, and **maximum tonal precision** for VFX, film, compositing, and color grading workflows.

---

## Quick Start

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Jorge-Studio/js-exr-upbitrate.git
cd js-exr-upbitrate
git checkout layering-test
pip install -r requirements.txt
# Restart ComfyUI
```

Then drag any workflow from `workflows/` into ComfyUI.

---

## 10 Curated Workflows (Simple → Powerful)

Start with 01 and work your way up. Each builds on the previous.

### Level 1: Beginner

| # | Workflow | Nodes | What It Does |
|---|----------|-------|-------------|
| 01 | `01_Quick_EXR_Export.json` | 3 | Drop in a PNG/JPG → get a 32-bit linear EXR with debanding. That's it. |
| 02 | `02_Color_Grade_Curves.json` | 8 | Full grading: exposure, contrast, lift/gamma/gain, Lightroom-style curves → EXR + sRGB preview |

### Level 2: Intermediate

| # | Workflow | Nodes | What It Does |
|---|----------|-------|-------------|
| 03 | `03_Fractal_BitDepth_Simple.json` | 7 | Fractal math fills 8-bit tonal gaps → 32-bit. See fractal map + waveform QC. No AI needed. |
| 04 | `04_Exposure_Brackets_HDR.json` | 8 | Generate 5 exposure brackets (+4 to -4 EV) from one image, preview each, save 16-bit TIFFs |
| 05 | `05_Shadow_HDR_Rec709.json` | 8 | Power-curve shadow control, proper Rec.709 via colour-science → EXR + preview |

### Level 3: Advanced

| # | Workflow | Nodes | What It Does |
|---|----------|-------|-------------|
| 06 | `06_Video_Deflicker_Export.json` | 7 | Load EXR sequence → luminance + normals deflicker → dual export (LogC3 + Rec.709 preview) |
| 07 | `07_AI_AutoSegment_Fractal.json` | 11 | **SAM 2.1 auto-segments** scene → fractal expansion per layer → composite → EXR. One click. |

### Level 4: Pro

| # | Workflow | Nodes | What It Does |
|---|----------|-------|-------------|
| 08 | `08_PerLayer_Edit_Pipeline.json` | 20 | Select individual layers, override labels, tune fractal/smooth per layer, enhance detail → EXR |
| 09 | `09_Cinema_Delivery_Package.json` | 7 | Full Molinare spec: ACES EXR sequence + CSV manifest + Rec.709 reference QuickTime |
| 10 | `10_Ultimate_Pipeline.json` | 18 | Everything combined: shadow control + AI segmentation + fractal + grading + curves + color match + QC |

All workflows are validated and tested against a running ComfyUI instance.

---

## SAM 2.1 Tiered Segmentation System

The Scene Segmenter uses a 4-tier detection system. Set `detection_backend` to `"auto"` and it picks the best available.

| Tier | Backend | Quality | Requires | Use Case |
|------|---------|---------|----------|----------|
| **1: sam3** | SAM 2.1 auto-mask generation (HuggingFace) | Excellent | `transformers>=4.40.0`, GPU, ~900MB model | Default for most users |
| **2: dinox_sam3** | DINO-X cloud detection + SAM 2.1 local masks | Best detection accuracy (59.8 AP LVIS) | DINO-X API key + internet | Maximum precision |
| **3: gdino_sam3** | Grounding DINO 1.0 + SAM 2.1 masks | Very good — proven local detector | `groundingdino` package | Fully offline, legacy |
| **4: fallback** | Luminance/edge thresholds | Basic | No AI, CPU only | Environments without GPU |

### Key Features

- **Auto-Describe Mode**: Enable `auto_describe` to automatically detect and label scene elements (sky, ground, person, trees, etc.) without typing prompts
- **Dynamic Layer Count**: No fixed cap — adapts to the scene (3 masks for a simple sky+ground, 15+ for a complex street scene)
- **Pipeline Caching**: SAM 2.1 model loads once and stays in GPU memory for fast subsequent runs
- **Graceful Fallback**: If SAM fails, automatically falls back to luminance-based segmentation

### Models Used

| Model Size | HuggingFace ID | Parameters | Speed |
|-----------|---------------|------------|-------|
| `large` (default) | `facebook/sam2.1-hiera-large` | 217M | Best quality |
| `base_plus` | `facebook/sam2.1-hiera-base-plus` | ~100M | Good balance |
| `tiny` | `facebook/sam2.1-hiera-tiny` | ~40M | Fastest |

### DINO-X API Setup (Tier 2)

For maximum detection accuracy:

1. Get a free API key at https://cloud.deepdataspace.com
2. `pip install dds-cloudapi-sdk`
3. In SceneSegmenter, set `detection_backend` to `"dinox_sam3"` and paste your key in `dinox_api_key`

---

## All 37 Nodes

### Core Color & EXR (10 nodes)

| Node | What It Does |
|------|-------------|
| **Prepare Image High Bit Depth** | sRGB → Linear + headroom + debanding |
| **Color Grading Controller** | Exposure, contrast, lift/gamma/gain, saturation |
| **HDR Curve Editor** | Lightroom-style parametric curve (shadows/mids/highlights/whites/blacks) |
| **Color Match to Reference** | Auto-match processed image colors to original |
| **Advanced Color Match** | 5 algorithms: Histogram, LAB, Reinhard, CLAHE, CDF |
| **Auto Exposure Match** | Quick exposure-only brightness matching |
| **Color Space Converter** | sRGB ↔ Linear ↔ ARRI LogC3 ↔ S-Log3 ↔ V-Log |
| **Save Image EXR** | Export 16/32-bit EXR with Log format options |
| **Save Video EXR Sequence** | Export video as numbered EXR sequence |
| **Image Stats** | Print range, unique values, effective bit depth |

### Fractal Bit-Depth (2 nodes)

| Node | What It Does |
|------|-------------|
| **Fractal Bit-Depth Expander** | 8→32-bit via Local Fractal Dimension, fBm, Hermite splines |
| **Perceptual Dither** | Blue noise / TPDF / fractal dither for banding-free output |

### AI Scene Segmentation (5 nodes)

| Node | What It Does |
|------|-------------|
| **Scene Segmenter (SAM 2.1 Tiered)** | Auto-detect & segment scene into semantic layers |
| **Layer Selector** | Pick one layer by index from the dynamic mask list |
| **Layer Decomposer** | Split image into individual layers using masks |
| **Segmentation Preview** | Color-coded overlay with labels, confidence, area% |
| **Layer Inpaint Prepare** | Prepare layer for inpainting with auto-generated prompts |

### Layer Processing (5 nodes)

| Node | What It Does |
|------|-------------|
| **Layer Fractal Processor** | Per-layer fractal expansion with semantic auto-tuning |
| **Layer Detail Editor** | Override labels, adjust fractal/smooth multipliers, skip layers |
| **Batch Layer Fractal Processor** | One-click: process ALL layers at once |
| **Layer Detail Enhancer** | Laplacian pyramid detail enhancement / sharpening |
| **Layer Assembler** | Composite all layers with feathered masks + colorspace conversion |

### Validation (1 node)

| Node | What It Does |
|------|-------------|
| **Bit-Depth Validator** | QC: unique values, PSNR, SSIM, gradient smoothness, waveform |

### Shadow-Controlled HDR (3 nodes)

| Node | What It Does |
|------|-------------|
| **Shadow Controlled Exposure** | 5 exposure brackets with power-curve shadow control |
| **Shadow Curve Processor** | Power-curve shadow noise suppression (not logarithmic) |
| **Rec.709 Converter** | Accurate colorspace conversion via `colour-science` |

### Exposure Bracketing (3 nodes)

| Node | What It Does |
|------|-------------|
| **Exposure Bracket Generator** | Generate 5 EV brackets (+4, +2, 0, -2, -4) from one image |
| **Exposure Bracket to TIFF** | Save brackets as 16-bit TIFF sequences |
| **Video to Exposure Brackets** | Load video/EXR/image sequence and bracket it |

### Cinema Delivery (3 nodes)

| Node | What It Does |
|------|-------------|
| **Save EXR Sequence (Cinema)** | DCI 4K, ACES 2065-1, PIZ — Molinare-spec delivery |
| **Generate Delivery CSV** | Professional CSV manifest |
| **ACES to Rec.709 Preview** | View transform for SDR monitoring |

### Animation & Motion (5 nodes)

| Node | What It Does |
|------|-------------|
| **Animated Pan & Scan** | Keyframe pan/zoom/rotation with easing |
| **Load EXR Image** | Load a single HDR EXR file |
| **Load EXR Sequence** | Load EXR sequence as video batch |
| **Extract Motion from Video** | Optical flow motion extraction |
| **Apply Motion Path** | Apply motion path to image sequences |

### Deflicker (2 nodes)

| Node | What It Does |
|------|-------------|
| **Luminance Deflicker** | Fix per-frame brightness without blur |
| **Normals Deflicker** | Stabilize gradients while preserving luminance |

---

## Semantic Auto-Tuning

When the fractal processor detects what each layer contains, it applies optimized presets:

| Layer Type | Octaves | Persistence | Smoothness | Strategy |
|-----------|---------|-------------|------------|----------|
| Sky | 2 | 0.30 | 0.95 | Ultra-smooth gradient fill |
| Cloud | 3 | 0.45 | 0.75 | Soft organic edges |
| Skin / Face | 3 | 0.35–0.40 | 0.70–0.75 | Subsurface-aware, no artifacts |
| Foliage / Trees | 6 | 0.60 | 0.30 | Rich organic micro-texture |
| Ground | 5 | 0.50 | 0.40 | Medium earth texture |
| Building | 3 | 0.35 | 0.60 | Sharp edges, smooth surfaces |
| Water | 4 | 0.45 | 0.80 | Smooth with ripple texture |
| Hair | 6 | 0.55 | 0.25 | Strand-level detail |

---

## Dependencies

```bash
pip install -r requirements.txt
```

| Package | Required | Purpose |
|---------|----------|---------|
| numpy | Yes | Core math |
| scipy | Yes | Gaussian filter, morphology, LFD |
| pillow | Yes | Image handling, font rendering |
| torch | Yes | Tensor ops (bundled with ComfyUI) |
| openexr, imath | Yes | 32-bit EXR writing |
| pyexr | Yes | EXR reading/writing |
| imageio | Recommended | EXR fallback + video |
| opencv-python | Recommended | Debanding, rotation, optical flow |
| colour-science | Recommended | Accurate Rec.709/ACES transforms |
| **transformers>=4.40.0** | **Recommended** | **SAM 2.1 segmentation (Tier 1)** |

### Optional

| Package | Purpose |
|---------|---------|
| `dds-cloudapi-sdk` | DINO-X cloud detection (Tier 2) |
| `groundingdino` | Legacy Grounding DINO (Tier 3) |

---

## Molinare / Professional DI Delivery

Supported specs: DCI 4K (4096x2160), ACES 2065-1, 16-bit half-float, PIZ compression, numbered EXR sequences, CSV manifests. Use workflow `09_Cinema_Delivery_Package.json`.

---

## Troubleshooting

### Scene Segmenter shows "fallback" only
Install `transformers>=4.40.0` for SAM 2.1 (Tier 1). On RunPod, pre-download the model to your network volume:

```bash
export HF_HOME=/workspace/models/sam2
python -c "from transformers import Sam2Model, Sam2Processor; Sam2Model.from_pretrained('facebook/sam2.1-hiera-large'); Sam2Processor.from_pretrained('facebook/sam2.1-hiera-large')"
```

### Red borders on nodes in ComfyUI
Make sure you're using the workflows from this branch — parameter values must match exactly (e.g. compression `piz` not `PIZ`).

### Cannot load EXR
`pip install pyexr openexr imath`

### colour-science API errors
`pip install --upgrade colour-science`

### Color looks wrong in Resolve
Verify Input Color Space matches export format. For ACES 2065-1 output, set Input to "ACES 2065-1 (AP0)" in Resolve.

---

## File Structure

```
js-exr-upbitrate/
  __init__.py                 # Core nodes + registration (37 nodes)
  fractal_utils.py            # Fractal math (LFD, fBm, Hermite, blue noise)
  fractal_bitdepth.py         # FractalBitDepthExpander, PerceptualDither
  scene_segmentation.py       # SceneSegmenter (SAM 2.1 tiered), LayerSelector,
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
  requirements.txt
  workflows/                  # 10 curated + additional workflow JSONs
  js/                         # ComfyUI web extensions
```

---

## License

MIT

## Links

- **GitHub**: https://github.com/Jorge-Studio/js-exr-upbitrate
- **ComfyUI**: https://github.com/comfyanonymous/ComfyUI

## Credits

Developed by **Jorge Studio / KS Films** for professional AI-to-cinema pipelines. Part of the **NodyJS** ecosystem.
