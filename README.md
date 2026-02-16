# High Bit Depth EXR Export & Cinema Delivery for ComfyUI (v3.1)

Professional-grade EXR export with **Log format support**, **color grading controls**, **cinema delivery compliance**, **motion animation**, **luminance-preserving deflicker**, and **maximum tonal precision** for VFX, film, compositing, and color grading workflows.

**Now includes full Molinare/Professional DI delivery support + blur-free deflicker!**

---

## Quick Start

1. Install the node (see Installation below)
2. Load any included workflow from `workflows/`
3. Connect your image and run!

**Recommended workflows:**
- `ultimate_exr_workflow.json` - Complete HDR processing pipeline
- `Test6_Cinema_HDR_Max.json` - Maximum quality cinema delivery
- `Video_ProRes4444_LogC3.json` - Video export in ProRes 4444

---

## What's New in v3.1

### Luminance-Preserving Deflicker (NEW!)
- **Luminance Deflicker**: Fix frame-to-frame brightness flicker **without blur**
  - Uses per-frame gain correction instead of pixel averaging
  - Methods: `gain_only`, `gain_and_offset`, `histogram_match`
  - Zero spatial processing = zero blur
- **Normals Deflicker**: Stabilize gradients/surface detail while preserving luminance
  - Works on image gradients (edges, textures)
  - Temporally smooth micro-flicker without affecting sharpness

### Cinema Delivery Nodes (Molinare Compliant)
- **Save EXR Sequence**: Export video frames as numbered EXR sequence with ACES 2065-1 color space
- **Generate Delivery CSV**: Create professional manifest files for post-house delivery
- **ACES → Rec.709 Preview**: View-transform for monitoring HDR content

### Animation & Motion Control
- **Animated Pan & Scan**: Keyframe-based pan, zoom, and rotation animation
- **Load EXR Image**: Load single HDR EXR files (ComfyUI default loader doesn't support EXR)
- **Load EXR Sequence**: Load EXR image sequences as video batches
- **Extract Motion from Video**: Optical flow-based motion path extraction
- **Apply Motion Path**: Apply extracted or manual motion to sequences

### Previous Features (v2.0)
- **Log Format Export**: ARRI LogC3, Sony S-Log3, Panasonic V-Log, Canon Log 3, RED Log3G10, DaVinci Intermediate
- **Color Space Converter**: Convert between sRGB, Linear, and Log formats
- **Color Grading Controller**: Professional lift/gamma/gain, exposure, contrast, saturation with **live preview**
- **HDR Curve Editor**: Lightroom-style shadows/midtones/highlights controls
- **Color Match to Reference**: Automatically match processed image to original
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
| imageio | Recommended | EXR fallback + video handling |
| opencv-python | Recommended | Debanding, rotation, optical flow |
| scipy | Optional | Motion smoothing |
| pillow | Recommended | Image resizing |

---

## All Nodes (20 Total)

### Core Color & EXR Nodes (10)

| Node | Description |
|------|-------------|
| **Prepare Image High Bit Depth** | sRGB→Linear + headroom + debanding |
| **Color Grading Controller** | Exposure, contrast, lift/gamma/gain, saturation |
| **HDR Curve Editor** | Shadows, midtones, highlights, whites, blacks |
| **Color Match to Reference** | Auto-match processed image colors to original |
| **Advanced Color Match** | 5 algorithms: Histogram, LAB, Reinhard, CLAHE, CDF |
| **Auto Exposure Match** | Quick exposure-only matching |
| **Color Space Converter** | Convert between sRGB, Linear, and Log formats |
| **Save Image EXR** | Export single image as 16/32-bit EXR with Log format |
| **Save Video EXR Sequence** | Export video as EXR sequence with Log format |
| **Image Stats** | Display range, unique values, bit depth estimate |

### Cinema Delivery Nodes (3)

| Node | Description |
|------|-------------|
| **Save EXR Sequence (Cinema)** | DCI 4K ACES 2065-1 EXR sequence for post-house delivery |
| **Generate Delivery CSV** | Professional manifest with shot info, frame ranges, specs |
| **ACES → Rec.709 Preview** | RRT+ODT view transform for preview monitoring |

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

## Molinare/Professional DI Delivery

This package is designed to meet professional post-production delivery specifications:

### Supported Specifications
- **Resolution**: DCI 4K (4096×2160), UHD 4K (3840×2160), 2K, 1080p
- **Color Space**: ACES 2065-1 (AP0), ACEScct, Linear Rec.709
- **Bit Depth**: 16-bit half-float or 32-bit full float
- **Compression**: PIZ (recommended), ZIP, ZIPS, RLE, None
- **Frame Rate**: 24fps (or any custom)
- **Naming**: `shot_name_V###.####.exr`

### Cinema Delivery Workflow

```
[Video/Image Source]
    ↓
Prepare Image High Bit Depth (sRGB→Linear + headroom)
    ↓
Color Grading Controller (creative adjustments)
    ↓
Color Space Converter (Linear → keep or convert)
    ↓
Save EXR Sequence (Cinema)
    └─ shot_name: "KSA_001_010"
    └─ convert_to_aces: true
    └─ bit_depth: 16
    └─ compression: piz
    ↓
Generate Delivery CSV (metadata manifest)
    ↓
ACES → Rec.709 Preview (for QC viewing)
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
    ↓
Animated Pan & Scan
    └─ output_size: 4K DCI
    └─ start_x: -500, start_y: 0, start_zoom: 1.0
    └─ end_x: 500, end_y: 0, end_zoom: 1.2
    └─ easing: ease_in_out
    └─ rotation_start: 0, rotation_end: 5
    ↓
[Animated Output]
```

### Motion Extraction from Reference Video

```
[Reference Video with Camera Movement]
    ↓
Extract Motion from Video
    └─ sensitivity: 1.0
    └─ smoothing: 5
    ↓
[motion_path output]

[Target Image/Video]
    ↓
Apply Motion Path ← [motion_path]
    └─ scale_motion: 1.0
    └─ invert_motion: false (true = stabilization)
    ↓
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

Traditional deflicker averages pixel colors across frames → **blur**.

Our approach:
1. **Measure** what's varying (brightness level, gradient stability)
2. **Smooth** the measurement, not the pixels
3. **Apply** a correction factor per frame

Result: Every pixel keeps its original relationship to neighbors = **zero blur**.

### Deflicker Workflow

```
[EXR Sequence with Flicker]
    ↓
Load EXR Sequence
    ↓
Luminance Deflicker
    └─ method: gain_only
    └─ smoothing_window: 7
    └─ strength: 1.0
    ↓
[Optional: Normals Deflicker for texture]
    ↓
Color Space Converter (Linear → LogC3)
    ↓
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

## Included Workflows

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

### Molinare Delivery Workflow
| Workflow | Description |
|----------|-------------|
| `Molinare_Delivery_Workflow.json` | Complete spec-compliant delivery pipeline |

---

## Node Settings Reference

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
1. Import EXR → Set Input Color Space to match (e.g., ACES 2065-1)
2. Timeline Color Space: DaVinci Wide Gamut / ACEScct
3. Apply creative grade
4. Output: Your delivery format

### Nuke
1. Read node → Set colorspace to ACES 2065-1 or Linear
2. Work in scene-linear
3. Use OCIOColorSpace for view transforms

### After Effects
1. Import EXR sequence
2. Interpret Footage → Color Management → Input Profile: ACES
3. Apply OCIO or manual LUT for viewing

### Baselight
1. Import EXR sequence
2. Set Input Colour Space: ACES 2065-1
3. Working Space: ACEScct (as per Molinare spec)

---

## Quality Verification

Use the **Image Stats** node to verify output quality:

```
[Your Processed Image] → Image Stats
```

Expected output for HDR:
- Range includes values > 1.0
- Unique values: 30,000+ (high precision)
- Effective bit depth: 14-16 bits

---

## Troubleshooting

### "Cannot load EXR" error
Install pyexr: `pip install pyexr`

### Motion nodes not showing rotation
Ensure opencv-python is installed: `pip install opencv-python`

### EXR sequence not loading correctly
- Use `start_frame: 0` to load all frames from beginning
- Check that frame numbers are in filename (e.g., `shot.1001.exr`)

### Color looks wrong in Resolve
- Verify Input Color Space matches export format
- For ACES 2065-1 output, set Input to "ACES 2065-1 (AP0)"

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
