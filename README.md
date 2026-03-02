# High Bit Depth EXR Export & Cinema Delivery for ComfyUI (v4.0)

Professional-grade EXR export with **Fractal Bit-Depth Expansion**, **AI Scene Segmentation**, **Log format support**, **color grading controls**, **cinema delivery compliance**, **motion animation**, **luminance-preserving deflicker**, and **maximum tonal precision** for VFX, film, compositing, and color grading workflows.

**v4.0: Fractal Bit-Depth Expansion + AI Scene Segmentation + Layer-Based Detail Enhancement**

---

## Quick Start

1. Install the node (see Installation below)
2. Load any included workflow from `workflows/`
3. Connect your image and run!

**Recommended workflows:**
- `Fractal_BitDepth_NoAI.json` - Fractal-only 8-to-32-bit expansion (no AI models needed)
- `Fractal_BitDepth_Full.json` - Full pipeline with AI segmentation + per-layer processing
- `Segmentation_Preview_Inpaint.json` - Segmentation preview + text-to-image inpainting setup
- `ultimate_exr_workflow.json` - Complete HDR processing pipeline
- `Test6_Cinema_HDR_Max.json` - Maximum quality cinema delivery
- `Video_ProRes4444_LogC3.json` - Video export in ProRes 4444

---

## What's New in v4.0

### Fractal Bit-Depth Expansion (NEW!)

Expand 8-bit source material to genuine 32-bit float using fractal mathematics:

- **Fractal Bit-Depth Expander**: Uses Local Fractal Dimension (LFD) analysis, fractal Brownian motion (fBm), Hermite spline interpolation, and rational fractal cubic splines to fill the tonal space between 8-bit quantization steps with perceptually correct data.
- **Perceptual Dither**: Apply blue noise, TPDF, or fractal dither patterns for banding-free gradients.

### AI Scene Segmentation (NEW!)

Decompose images into semantic regions for per-layer processing:

- **Scene Segmenter (AI)**: Uses SAM2 + GroundingDINO for text-guided segmentation or luminance/edge fallback when AI models are unavailable. Generates depth maps via Depth Anything V2.
- **Layer Decomposer**: Extracts individual layers with alpha masks and per-layer statistics (area, luminance, fractal dimension).
- **Segmentation Preview**: Color-coded visualization of all segments with labels, area percentages, and LFD values overlaid.
- **Layer Inpaint Prepare**: Sets up each segment for text-to-image inpainting through ComfyUI's KSampler. Built-in prompt templates for sky, foliage, ground, building, person, water, skin, etc. across photorealistic/cinematic/natural styles.

### Layer Processing Pipeline (NEW!)

- **Layer Fractal Processor**: Applies fractal bit-depth expansion per-layer with auto-tuned parameters based on semantic labels (sky, foliage, skin, etc. each get optimized settings).
- **Layer Detail Enhancer**: Frequency-domain blending via Laplacian pyramid to inject AI-generated or synthetic micro-detail while preserving original color grading.
- **Layer Assembler**: Composites processed layers back with alpha feathering and optional ACES 2065-1 conversion.
- **Bit-Depth Validator**: QC node reporting unique values, PSNR, SSIM, gradient smoothness, and waveform visualization.

### Shadow-Controlled HDR (NEW!)

- **Shadow Controlled Exposure**: Power-curve shadow control with noise suppression.
- **Shadow Curve Processor**: Non-linear curves that peak instead of roll off for tonal precision.
- **Rec.709 Converter**: Accurate Rec.709 conversion using the `colour-science` library.

### Exposure Bracketing (NEW!)

- **Exposure Bracket Generator**: Generate multiple EV levels from a single source for HDR merging.
- **Exposure Bracket to TIFF**: Save 16-bit TIFF brackets for archival.
- **Video to Exposure Brackets**: Process video into bracket sequences for HDR pipelines.

---

## What's in v3.x

### Luminance-Preserving Deflicker
- **Luminance Deflicker**: Fix frame-to-frame brightness flicker **without blur**
  - Uses per-frame gain correction instead of pixel averaging
  - Methods: `gain_only`, `gain_and_offset`, `histogram_match`
- **Normals Deflicker**: Stabilize gradients/surface detail while preserving luminance

### Cinema Delivery Nodes (Molinare Compliant)
- **Save EXR Sequence**: Export video frames as numbered EXR sequence with ACES 2065-1
- **Generate Delivery CSV**: Create professional manifest files
- **ACES to Rec.709 Preview**: View-transform for monitoring HDR content

### Animation & Motion Control
- **Animated Pan & Scan**: Keyframe-based pan, zoom, and rotation animation
- **Load EXR Image / Sequence**: Load HDR EXR files
- **Extract Motion from Video**: Optical flow-based motion path extraction
- **Apply Motion Path**: Apply extracted or manual motion to sequences

### Core Color & EXR
- **Log Format Export**: ARRI LogC3, Sony S-Log3, Panasonic V-Log, Canon Log 3, RED Log3G10, DaVinci Intermediate
- **Color Space Converter**: Convert between sRGB, Linear, and Log formats
- **Color Grading Controller**: Professional lift/gamma/gain, exposure, contrast, saturation
- **HDR Curve Editor**: Lightroom-style shadows/midtones/highlights controls
- **Color Match to Reference**: Auto-match processed image to original
- **Advanced Color Match**: 5 professional color calibration algorithms
- **Auto Exposure Match**: Quick brightness matching with exposure stop readout
- **Image Stats**: Verify range, unique values, and effective bit depth

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
| torch | **Yes** | Tensor operations (bundled with ComfyUI) |

#### Optional AI Models (for Scene Segmenter)

| Model | Purpose | Auto-download |
|-------|---------|---------------|
| SAM2 (segment-anything-2) | Semantic segmentation | Manual install |
| GroundingDINO | Text-guided object detection | Manual install |
| Depth Anything V2 | Monocular depth estimation | Manual install |

The Scene Segmenter falls back to luminance/edge-based segmentation if AI models are not installed.

---

## All Nodes (35 Total)

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
| **Fractal Bit-Depth Expander** | 8-to-32-bit expansion via LFD, fBm, Hermite splines, and RFC interpolation |
| **Perceptual Dither** | Blue noise, TPDF, or fractal dither for banding-free output |

### AI Scene Segmentation Nodes (4)

| Node | Description |
|------|-------------|
| **Scene Segmenter (AI)** | Text-guided semantic segmentation with SAM2/GroundingDINO or luminance fallback |
| **Layer Decomposer** | Extract layers with alpha masks, per-layer stats (area, luminance, LFD) |
| **Segmentation Preview** | Color-coded overlay visualization of all segments with labels |
| **Layer Inpaint Prepare** | Prepare per-segment image + mask + prompt for KSampler inpainting |

### Layer Processing Nodes (4)

| Node | Description |
|------|-------------|
| **Layer Fractal Processor** | Per-layer fractal expansion with semantic-aware auto-tuning |
| **Layer Detail Enhancer** | Laplacian pyramid frequency blending for micro-detail injection |
| **Layer Assembler** | Composite layers with alpha feathering + optional ACES conversion |
| **Bit-Depth Validator** | QC: unique values, PSNR, SSIM, gradient smoothness, waveform |

### Shadow-Controlled HDR Nodes (3)

| Node | Description |
|------|-------------|
| **Shadow Controlled Exposure** | Power-curve shadow control with noise suppression |
| **Shadow Curve Processor** | Non-linear curves that peak instead of rolling off |
| **Rec.709 Converter** | Accurate Rec.709 via `colour-science` library |

### Exposure Bracketing Nodes (3)

| Node | Description |
|------|-------------|
| **Exposure Bracket Generator** | Generate multiple EV brackets from single source |
| **Exposure Bracket to TIFF** | Save 16-bit TIFF exposure brackets |
| **Video to Exposure Brackets** | Process video into bracket sequences |

### Cinema Delivery Nodes (3)

| Node | Description |
|------|-------------|
| **Save EXR Sequence (Cinema)** | DCI 4K ACES 2065-1 EXR sequence for post-house delivery |
| **Generate Delivery CSV** | Professional manifest with shot info, frame ranges, specs |
| **ACES to Rec.709 Preview** | RRT+ODT view transform for preview monitoring |

### Animation & Motion Nodes (5)

| Node | Description |
|------|-------------|
| **Animated Pan & Scan** | Keyframe animation for pan, zoom, rotation with easing |
| **Load EXR Image** | Load single HDR EXR files |
| **Load EXR Sequence** | Load EXR sequence as video batch |
| **Extract Motion from Video** | Optical flow motion path extraction |
| **Apply Motion Path** | Apply motion to image/video sequences |

### Deflicker Nodes (2)

| Node | Description |
|------|-------------|
| **Luminance Deflicker** | Per-frame gain correction for brightness flicker (NO BLUR) |
| **Normals Deflicker** | Gradient-preserving temporal smoothing (preserves luminance) |

---

## Fractal Bit-Depth Expansion

### How It Works

The core innovation is using fractal mathematics to fill the 24 empty bit-levels between 8-bit (256 values) and 32-bit float (16M+ values):

1. **Local Fractal Dimension (LFD)**: Analyzes image complexity at every pixel using box-counting. High LFD (edges, texture) gets fractal noise; low LFD (sky, gradients) gets smooth Hermite interpolation.

2. **Fractal Brownian Motion (fBm)**: Generates micro-texture that looks natural at all scales, injected into detailed regions at the sub-pixel level.

3. **Hermite Spline Interpolation**: Smoothly fills tonal steps in flat regions (sky, skin) without introducing artifacts.

4. **Rational Fractal Cubic (RFC) Spline**: Blends cubic interpolation with fractal perturbation for medium-complexity areas.

5. **Blue Noise / TPDF Dither**: Perceptually optimized noise patterns break remaining banding.

### Fractal-Only Pipeline (No AI Needed)

```
[8-bit Source Image/Video]
    |
Fractal Bit-Depth Expander
    |-- fractal_strength: 0.3 (how much fractal detail)
    |-- smooth_strength: 0.7 (how smooth flat regions)
    |-- lfd_window: 9 (analysis window size)
    |
Perceptual Dither
    |-- method: blue_noise
    |-- strength: 0.3
    |
[32-bit Float Output]
    |
Save Image EXR (32-bit, LogC3)
```

### Full Pipeline with AI Segmentation

```
[8-bit Source Image]
    |
Scene Segmenter (AI)
    |-- labels: "sky, trees, ground, building, person"
    |-- method: sam2_grounding_dino (or luminance_edge fallback)
    |
    +--> Segmentation Preview (visualize all layers)
    |
    +--> Layer Decomposer (extract layers)
           |
           +--> Layer Fractal Processor (sky -- smooth, low fractal)
           +--> Layer Fractal Processor (trees -- high fractal, texture)
           +--> Layer Fractal Processor (ground -- medium fractal)
           +--> Layer Detail Enhancer (per-layer sharpening)
           |
           +--> Layer Assembler (composite all layers)
                  |
                  Bit-Depth Validator (QC report)
                  |
                  Save Image EXR (32-bit ACES)
```

### Segmentation Preview & Inpaint Pipeline

```
[Source Image]
    |
Scene Segmenter (AI)
    |-- labels: "sky, trees, ground, person"
    |
    +--> Segmentation Preview
    |      |-- overlay_opacity: 0.45
    |      |-- show_labels: true
    |      +--> PreviewImage (color-coded layer map)
    |
    +--> Layer Inpaint Prepare (sky)
    |      |-- layer_label: "sky"
    |      |-- detail_prompt_style: cinematic
    |      |-- expand_mask_px: 10
    |      +--> inpaint_image --> SetLatentNoiseMask --> KSampler
    |      +--> suggested_prompt --> CLIP Text Encode
    |
    +--> Layer Inpaint Prepare (ground)
           |-- layer_label: "ground"
           +--> [same KSampler chain]
```

### Layer Inpaint Prepare -- Built-in Prompt Templates

| Layer Label | Cinematic Prompt |
|-------------|-----------------|
| sky | cinematic sky, atmospheric perspective, film grain, anamorphic |
| cloud | dramatic cloud formations, golden hour light, cinematic |
| trees / foliage | cinematic vegetation, volumetric light through leaves |
| ground | cinematic ground plane, shallow depth of field |
| person | cinematic portrait lighting, film emulation, anamorphic bokeh |
| building | cinematic architecture, dramatic lighting, production design |
| water | cinematic water, light play on surface, anamorphic |
| skin / face | cinematic skin tones, beauty lighting, film emulation |

Also available in `photorealistic` and `natural` styles, or use `custom` with your own prompt.

### Semantic-Aware Auto-Tuning (Layer Fractal Processor)

The Layer Fractal Processor automatically adjusts parameters based on what the layer contains:

| Layer Type | Fractal Strength | Smooth Strength | Strategy |
|-----------|-----------------|-----------------|----------|
| Sky | Low (0.15) | High (0.85) | Smooth gradients, minimal noise |
| Foliage | High (0.55) | Low (0.35) | Rich micro-texture |
| Skin / Face | Very Low (0.10) | Very High (0.90) | Ultra-smooth, no artifacts |
| Ground | Medium (0.35) | Medium (0.55) | Balanced detail |
| Building | Medium-High (0.40) | Medium (0.50) | Structural detail |
| Water | Low-Medium (0.25) | High (0.70) | Smooth with caustic texture |

### Quality Metrics (Bit-Depth Validator)

| Metric | 8-bit Input | After Fractal Expansion | Target |
|--------|------------|------------------------|--------|
| Unique Values | ~220 | 50,000+ | >30,000 |
| PSNR vs Original | -- | >40 dB | >35 dB |
| SSIM vs Original | -- | >0.98 | >0.95 |
| Gradient Smoothness | Low | High | High |

---

## Molinare/Professional DI Delivery

This package is designed to meet professional post-production delivery specifications:

### Supported Specifications
- **Resolution**: DCI 4K (4096x2160), UHD 4K (3840x2160), 2K, 1080p
- **Color Space**: ACES 2065-1 (AP0), ACEScct, Linear Rec.709
- **Bit Depth**: 16-bit half-float or 32-bit full float
- **Compression**: PIZ (recommended), ZIP, ZIPS, RLE, None
- **Frame Rate**: 24fps (or any custom)
- **Naming**: `shot_name_V###.####.exr`

### Cinema Delivery Workflow

```
[Video/Image Source]
    |
Prepare Image High Bit Depth (sRGB -> Linear + headroom)
    |
Color Grading Controller (creative adjustments)
    |
Color Space Converter (Linear -> keep or convert)
    |
Save EXR Sequence (Cinema)
    |-- shot_name: "KSA_001_010"
    |-- convert_to_aces: true
    |-- bit_depth: 16
    |-- compression: piz
    |
Generate Delivery CSV (metadata manifest)
    |
ACES to Rec.709 Preview (for QC viewing)
```

### Example Manifest Output

```csv
Field,Value
Shot Name,KSA_001_010_V001
Resolution,4096x2160
Color Space,ACES 2065-1
Framerate,24 fps
Frame Range,1001-1120
Total Frames,120
Duration,5.00 seconds
Format,OpenEXR 16-bit half-float
Compression,PIZ
Delivery Date,2026-01-26 14:30
Generated By,NodyJS/ComfyUI Cinema Delivery
```

---

## Animation & Motion Control

### Manual Keyframe Animation

Use **Animated Pan & Scan** for cinematic camera movements:

```
[Large Source Image or Video]
    |
Animated Pan & Scan
    |-- output_size: 4K DCI
    |-- start_x: -500, start_y: 0, start_zoom: 1.0
    |-- end_x: 500, end_y: 0, end_zoom: 1.2
    |-- easing: ease_in_out
    |-- rotation_start: 0, rotation_end: 5
    |
[Animated Output]
```

### Motion Extraction from Reference Video

```
[Reference Video with Camera Movement]
    |
Extract Motion from Video
    |-- sensitivity: 1.0
    |-- smoothing: 5
    |
[motion_path output]

[Target Image/Video]
    |
Apply Motion Path <-- [motion_path]
    |-- scale_motion: 1.0
    |-- invert_motion: false (true = stabilization)
    |
[Output with Matching Motion]
```

### Easing Functions

| Easing | Description |
|--------|-------------|
| `linear` | Constant speed |
| `ease_in` | Start slow, accelerate |
| `ease_out` | Start fast, decelerate |
| `ease_in_out` | Smooth acceleration and deceleration |

---

## Deflicker Guide

### The Problem

AI-generated video often has frame-to-frame brightness or texture flicker that traditional temporal averaging cannot fix without causing blur.

### The Solution

Our deflicker nodes work **on measurements, not pixels**:

| Method | How It Works | Best For |
|--------|--------------|----------|
| **Luminance Deflicker (gain_only)** | Measures frame brightness, smooths the curve, applies gain correction | Brightness pumping, exposure flicker |
| **Luminance Deflicker (histogram_match)** | Matches each frame's histogram to a reference | Color/tonal inconsistency |
| **Normals Deflicker** | Smooths image gradients while preserving luminance | Texture flicker, grain dancing |

### Why No Blur?

Traditional deflicker averages pixel colors across frames -- **blur**.

Our approach:
1. **Measure** what's varying (brightness level, gradient stability)
2. **Smooth** the measurement, not the pixels
3. **Apply** a correction factor per frame

Result: Every pixel keeps its original relationship to neighbors = **zero blur**.

### Deflicker Workflow

```
[EXR Sequence with Flicker]
    |
Load EXR Sequence
    |
Luminance Deflicker
    |-- method: gain_only
    |-- smoothing_window: 7
    |-- strength: 1.0
    |
[Optional: Normals Deflicker for texture]
    |
Color Space Converter (Linear -> LogC3)
    |
VHS_VideoCombine (H.265/ProRes output)
```

---

## Log Format Guide

| Format | Use Case |
|--------|----------|
| **ARRI LogC3** | Industry standard, best for general use and Resolve |
| **Sony S-Log3** | Match Sony camera footage |
| **Panasonic V-Log** | Match Panasonic camera footage |
| **Canon Log 3** | Match Canon camera footage |
| **RED Log3G10** | Match RED camera footage |
| **DaVinci Intermediate** | Native DaVinci Resolve working space |
| **Linear** | VFX compositing in Nuke, After Effects |

---

## Included Workflows (25)

### Fractal & Segmentation Workflows
| Workflow | Description |
|----------|-------------|
| `Fractal_BitDepth_NoAI.json` | Fractal-only 8-to-32-bit expansion pipeline |
| `Fractal_BitDepth_Full.json` | Full pipeline: AI segmentation + per-layer fractal processing |
| `Segmentation_Preview_Inpaint.json` | Segmentation preview + text-to-image inpaint setup |

### Bit-Depth & HDR Workflows
| Workflow | Description |
|----------|-------------|
| `BitDepth_From_Single_Source.json` | Exposure bracketing + HDR merge from single source |
| `Anti_Flicker_HDR_Pipeline.json` | Anti-flicker HDR processing chain |

### Core Workflows
| Workflow | Description |
|----------|-------------|
| `ultimate_exr_workflow.json` | Complete HDR pipeline with all nodes |
| `advanced_color_match_workflow.json` | Compare all 5 color matching algorithms |

### Cinema Delivery Workflows
| Workflow | Description |
|----------|-------------|
| `Test1_HDR_Full_Pipeline.json` | HDR processing with debanding and grading |
| `Test2_ACES_Delivery.json` | Direct ACES 2065-1 EXR export |
| `Test3_Specular_Expand.json` | HDR highlight expansion |
| `Test4_ColorGrade_LogC3.json` | Professional color grading to ARRI LogC3 |
| `Test5_Rec709_Preview.json` | Rec.709 preview generation |
| `Test6_Cinema_HDR_Max.json` | Maximum quality cinema output |

### Video Export Workflows
| Workflow | Description |
|----------|-------------|
| `Video_ProRes4444_LogC3.json` | ProRes 4444 video export in LogC3 |
| `Video_H265_Rec709_Preview.json` | H.265 preview video for QC |

### Deflicker Workflows
| Workflow | Description |
|----------|-------------|
| `Luminance_Deflicker_NoBlur.json` | Brightness correction without blur |
| `Normals_Deflicker_GradPreserve.json` | Gradient-based flicker reduction |
| `Quick_Deflicker_Video.json` | Quick video deflicker pipeline |

### Molinare Delivery Workflow
| Workflow | Description |
|----------|-------------|
| `Molinare_Delivery_Workflow.json` | Complete spec-compliant delivery pipeline |

### Testing & Utility Workflows
| Workflow | Description |
|----------|-------------|
| `test_deband_effect.json` | Test debanding effect |
| `test_good_exr.json` | Test EXR output quality |
| `test_high_bit_image.json` | High bit-depth image test |
| `test_minimal_exr.json` | Minimal EXR export test |
| `test_replicate_good.json` | Replicate known-good output |

---

## Node Settings Reference

### Fractal Bit-Depth Expander

| Setting | Default | Description |
|---------|---------|-------------|
| fractal_strength | 0.3 | Amount of fractal micro-texture to inject (0-1) |
| smooth_strength | 0.7 | Hermite spline smoothness for flat regions (0-1) |
| lfd_window | 9 | Local Fractal Dimension analysis window (5-21) |
| temporal_coherence | true | Maintain consistency across video frames |

### Scene Segmenter (AI)

| Setting | Default | Description |
|---------|---------|-------------|
| labels | "sky, trees, ground, building, person, other" | Comma-separated semantic labels |
| method | luminance_edge | Segmentation method (sam2_grounding_dino or luminance_edge) |
| confidence | 0.5 | Detection confidence threshold (AI only) |
| num_layers | 6 | Maximum number of output layers |
| use_depth | false | Generate depth map for layer ordering |

### Segmentation Preview

| Setting | Default | Description |
|---------|---------|-------------|
| overlay_opacity | 0.45 | Color overlay strength (0.1-0.9) |
| show_labels | true | Draw labels at mask centroids |

### Layer Inpaint Prepare

| Setting | Default | Description |
|---------|---------|-------------|
| layer_label | "sky" | Semantic label for prompt template lookup |
| expand_mask_px | 10 | Dilate mask edges for smoother blending |
| detail_prompt_style | cinematic | Prompt style: photorealistic, cinematic, natural, custom |
| custom_prompt | "" | Override prompt (used when style is "custom") |

### Save EXR Sequence (Cinema)

| Setting | Default | Description |
|---------|---------|-------------|
| shot_name | KSA_001_010 | Shot identifier without version |
| version | 1 | Version number (becomes V001, V002, etc.) |
| start_frame | 1001 | Industry-standard start frame |
| input_is_linear | true | Input is scene-linear |
| convert_to_aces | true | Convert to ACES 2065-1 |
| bit_depth | 16 | Half-float for Molinare spec |
| compression | piz | PIZ recommended for delivery |

### Animated Pan & Scan

| Setting | Default | Description |
|---------|---------|-------------|
| output_size | 4K DCI | Output resolution preset |
| start_x/y | 0 | Starting pan position |
| start_zoom | 1.0 | Starting zoom (1.0 = 100%) |
| end_x/y | 0 | Ending pan position |
| end_zoom | 1.0 | Ending zoom level |
| easing | ease_in_out | Animation curve type |
| rotation_start/end | 0.0 | Rotation in degrees |
| loop_mode | none | Animation loop behavior |

### Color Grading Controller

| Setting | Range | Description |
|---------|-------|-------------|
| exposure | -5 to +5 | Exposure in stops |
| contrast | 0.5 to 2.0 | Contrast around 18% grey |
| lift | -0.5 to 0.5 | Shadow offset |
| gamma | 0.5 to 2.0 | Midtone gamma |
| gain | 0.5 to 2.0 | Highlight multiplier |
| saturation | 0 to 2.0 | Color saturation |

### Advanced Color Match

| Algorithm | Best For |
|-----------|----------|
| **Histogram Matching** | General-purpose, fast |
| **LAB Color Space** | Better perceptual color accuracy |
| **Reinhard Transfer** | Classic color transfer |
| **CLAHE + Histogram** | Local contrast + global color |
| **CDF Matching** | Precise statistical matching |

---

## Importing into Professional Software

### DaVinci Resolve
1. Import EXR -> Set Input Color Space to match (e.g., ACES 2065-1)
2. Timeline Color Space: DaVinci Wide Gamut / ACEScct
3. Apply creative grade
4. Output: Your delivery format

### Nuke
1. Read node -> Set colorspace to ACES 2065-1 or Linear
2. Work in scene-linear
3. Use OCIOColorSpace for view transforms

### After Effects
1. Import EXR sequence
2. Interpret Footage -> Color Management -> Input Profile: ACES
3. Apply OCIO or manual LUT for viewing

### Baselight
1. Import EXR sequence
2. Set Input Colour Space: ACES 2065-1
3. Working Space: ACEScct (as per Molinare spec)

---

## Quality Verification

Use the **Bit-Depth Validator** or **Image Stats** node to verify output quality:

```
[Your Processed Image] --> Bit-Depth Validator
```

Expected output for fractal-expanded HDR:
- Unique values: 50,000+ (genuine high bit-depth)
- PSNR vs original: >40 dB (faithful to source)
- SSIM: >0.98 (structurally identical)
- Gradient smoothness: High (no banding)

---

## File Structure

```
js-exr-upbitrate/
  __init__.py                 # Core nodes + registration
  fractal_utils.py            # Fractal math library (LFD, fBm, Hermite, blue noise)
  fractal_bitdepth.py         # FractalBitDepthExpander, PerceptualDither
  scene_segmentation.py       # SceneSegmenter, LayerDecomposer, SegmentationPreview, LayerInpaintPrepare
  layer_processor.py          # LayerFractalProcessor
  ai_detail_layer.py          # LayerDetailEnhancer
  layer_assembly.py           # LayerAssembler, BitDepthValidator
  cinema_delivery.py          # SaveEXRSequence, GenerateDeliveryCSV, ACESToRec709Preview
  animated_motion.py          # AnimatedPanAndScan, LoadEXR*, MotionPath*
  luminance_deflicker.py      # LuminanceDeflicker, NormalsDeflicker
  exposure_bracketing.py      # ExposureBracketGenerator, ExposureBracketToTIFF, VideoToExposureBrackets
  shadow_controlled_hdr.py    # ShadowControlledExposure, ShadowCurveProcessor, Rec709Converter
  requirements.txt            # Python dependencies
  workflows/                  # 25 ready-to-use workflow JSONs
  js/                         # ComfyUI web extensions
```

---

## Troubleshooting

### "Cannot load EXR" error
Install pyexr: `pip install pyexr`

### Scene Segmenter shows "luminance_edge" only
AI models (SAM2, GroundingDINO) not installed. The node still works using luminance/edge fallback.

### Motion nodes not showing rotation
Ensure opencv-python is installed: `pip install opencv-python`

### EXR sequence not loading correctly
- Use `start_frame: 0` to load all frames from beginning
- Check that frame numbers are in filename (e.g., `shot.1001.exr`)

### Color looks wrong in Resolve
- Verify Input Color Space matches export format
- For ACES 2065-1 output, set Input to "ACES 2065-1 (AP0)"

### colour-science API errors
Update to latest version: `pip install --upgrade colour-science`

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
