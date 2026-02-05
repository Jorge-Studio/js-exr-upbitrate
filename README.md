# High Bit Depth EXR Export for ComfyUI (v2.0)

Professional-grade EXR export with **Log format support**, **color grading controls**, **auto color matching**, and **maximum tonal precision** for VFX, compositing, and color grading workflows.

---

## Quick Start - Ultimate Workflow

**The fastest way to get the best results:**

1. Install the node (see Installation below)
2. Load the included workflow: `workflows/ultimate_exr_workflow.json`
3. Select your image and run!

**Also included:** `workflows/advanced_color_match_workflow.json` - Compare all 5 color matching algorithms side by side!

This workflow uses all 9 nodes in the optimal configuration for professional output.

---

## What's New in v2.0

- **Log Format Export**: ARRI LogC3, Sony S-Log3, Panasonic V-Log, Canon Log 3, RED Log3G10, DaVinci Intermediate
- **Color Space Converter**: Convert between sRGB, Linear, and Log formats
- **Color Grading Controller**: Professional lift/gamma/gain, exposure, contrast, saturation with **live preview**
- **HDR Curve Editor**: Lightroom-style shadows/midtones/highlights controls with **interactive curve display**
- **Color Match to Reference**: Automatically match processed image to original (prevents dark/shifted outputs)
- **Advanced Color Match**: 5 professional color calibration algorithms (Histogram, LAB, Reinhard, CLAHE, CDF)
- **Auto Exposure Match**: Quick brightness matching with exposure stop readout
- **Image Stats**: Verify range, unique values, and effective bit depth
- **Improved Precision**: Up to 42,000+ unique values vs ~4,000 in v1

---

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Jorge-Studio/js-exr-upbitrate.git
pip install openexr imath imageio opencv-python
# Restart ComfyUI
```

---

## Nodes Included (10 Total)

| Node | Description |
|------|-------------|
| **Prepare Image High Bit Depth** | sRGB→Linear + headroom + debanding |
| **Color Grading Controller** | Exposure, contrast, lift/gamma/gain, saturation (with live preview) |
| **HDR Curve Editor** | Shadows, midtones, highlights, whites, blacks (with curve display) |
| **Color Match to Reference** | Auto-match processed image colors/brightness to original |
| **Advanced Color Match** | 5 professional algorithms: Histogram, LAB, Reinhard, CLAHE, CDF |
| **Auto Exposure Match** | Quick exposure-only matching with stop readout |
| **Color Space Converter** | Convert between sRGB, Linear, and Log formats |
| **Save Image EXR** | Export single image as 16/32-bit EXR with Log format |
| **Save Video EXR Sequence** | Export video as EXR sequence with Log format |
| **Image Stats** | Display range, unique values, bit depth estimate |

---

## Ultimate Workflow (Recommended)

Load `workflows/ultimate_exr_workflow.json` for the complete pipeline:

```
1. LOAD IMAGE ─────────────────────────────────────────────────────────┐
       │                                                               │
       ↓                                                               │
2. PREPARE (sRGB→Linear + 0.5 stop headroom + subtle deband)           │
       │                                                               │
       ↓                                                               │
3. COLOR GRADING (exposure, contrast, lift/gamma/gain, saturation)     │
       │      [Live preview updates as you adjust!]                    │
       ↓                                                               │
4. HDR CURVES (fine-tune shadows, midtones, highlights)                │
       │      [Interactive curve visualization!]                       │
       ↓                                                               │
5. COLOR MATCH TO REFERENCE ←──────────────────────────────────────────┘
       │      (auto-matches brightness/contrast/colors to original)
       │      [Prevents dark or color-shifted outputs!]
       ↓
6. IMAGE STATS (verify quality metrics before export)
       │
       ↓
7. SAVE EXR (32-bit, ARRI LogC3, ZIP compression)
       │
       ├──→ 8. COLOR SPACE CONVERTER (Linear→sRGB) → PREVIEW
       │
       └──→ 9. AUTO EXPOSURE MATCH (alternative comparison)
```

### Why This Workflow Works Best

1. **Headroom**: 0.5 stops of highlight headroom gives colorists room to work
2. **Auto Color Match**: Ensures your graded image stays true to the original's brightness/colors
3. **Live Previews**: See changes before running the full workflow
4. **Quality Verification**: Image Stats confirms you're getting maximum precision
5. **Industry Standard Output**: 32-bit ARRI LogC3 works with any professional software

---

## Other Recommended Workflows

### For Color Grading (Matching Camera Footage)

```
[Image Source] 
    ↓
Prepare Image High Bit Depth
    └─ input_is_srgb: true
    └─ add_headroom: 0.5 (gives room for grading)
    └─ deband_strength: 0.3
    ↓
Save Image EXR
    └─ bit_depth: 32
    └─ output_format: ARRI LogC3 (or match your camera)
    └─ compression: zip
```

### For VFX Compositing (Linear Workflow)

```
[Image Source] 
    ↓
Color Space Converter
    └─ input_space: sRGB (ComfyUI Default)
    └─ output_space: Linear
    ↓
Save Image EXR
    └─ bit_depth: 32
    └─ output_format: Linear
```

### With Color Grading

```
[Image Source]
    ↓
Prepare Image High Bit Depth
    ↓
Color Grading Controller
    └─ exposure: 0.5
    └─ contrast: 1.1
    └─ saturation: 1.2
    ↓
HDR Curve Editor (optional)
    └─ shadows: 0.1
    └─ highlights: -0.05
    ↓
Save Image EXR (ARRI LogC3)
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

## Node Settings

### Prepare Image High Bit Depth

| Setting | Default | Description |
|---------|---------|-------------|
| `input_is_srgb` | true | Input is sRGB (standard ComfyUI output) |
| `add_headroom` | 0.0 | Add highlight headroom in stops (0.5-1.0 recommended) |
| `deband_strength` | 0.0 | Remove banding (0.3-0.5 subtle, 1.0+ aggressive) |

### Save Image EXR

| Setting | Default | Description |
|---------|---------|-------------|
| `bit_depth` | 32 | 16-bit (HALF) or 32-bit (FLOAT) |
| `compression` | zip | none, zip, rle, zips, piz, dwaa |
| `output_format` | ARRI LogC3 | Log format for export |
| `input_is_linear` | true | Input is linear (from Prepare node) |

### Color Grading Controller

| Setting | Range | Description |
|---------|-------|-------------|
| `exposure` | -5 to +5 | Exposure in stops |
| `contrast` | 0.5 to 2.0 | Contrast around 18% grey |
| `lift` | -0.5 to 0.5 | Shadow offset |
| `gamma` | 0.5 to 2.0 | Midtone gamma |
| `gain` | 0.5 to 2.0 | Highlight multiplier |
| `saturation` | 0 to 2.0 | Color saturation |

### HDR Curve Editor

| Setting | Range | Description |
|---------|-------|-------------|
| `blacks` | -1 to 1 | Black point adjustment |
| `shadows` | -1 to 1 | Shadow tones |
| `midtones` | -1 to 1 | Midtone adjustment |
| `highlights` | -1 to 1 | Highlight tones |
| `whites` | -1 to 1 | White point adjustment |

### Advanced Color Match

| Setting | Options | Description |
|---------|---------|-------------|
| `method` | Histogram Matching, LAB Color Space, Reinhard Transfer, CLAHE + Histogram, CDF Matching | Color calibration algorithm |
| `strength` | 0.0 to 1.0 | Blend strength (0=no change, 1=full match) |
| `match_luminance_only` | true/false | Only match brightness, preserve original colors |

**Algorithm Guide:**

| Algorithm | Best For |
|-----------|----------|
| **Histogram Matching** | General-purpose, fast, good for most images |
| **LAB Color Space** | Better perceptual color accuracy |
| **Reinhard Transfer** | Classic color transfer, preserves structure |
| **CLAHE + Histogram** | Local contrast + global color matching |
| **CDF Matching** | Precise statistical matching |

---

## Quality Comparison (v2 vs v1)

| Metric | v1 | v2 |
|--------|----|----|
| Unique values | ~4,000 | **42,000+** |
| Effective bit depth | ~12 bits | **~16 bits** |
| Log format support | No | **Yes (7 formats)** |
| Color grading | No | **Yes** |
| Headroom control | No | **Yes** |

---

## Importing into Professional Software

### DaVinci Resolve

1. Import EXR → Set Input Color Space to match export format (e.g., ARRI LogC3)
2. Timeline Color Space: DaVinci Wide Gamut / Intermediate
3. Output: Your delivery format

### After Effects

1. Import EXR sequence
2. Interpret Footage → Color Management → Input Profile: Match your Log format
3. Use OCIO or manual LUT for viewing

### Nuke

1. Read node → Set colorspace to match export format
2. Work in linear space
3. Use OCIOColorSpace nodes for conversions

---

## Verifying Output Quality

Use the **Image Stats** node to verify your output:

```
[Your Image] → Image Stats
```

Output shows:
- Range (should include values > 1.0 for HDR)
- Unique values (higher = more precision)
- Effective bit depth estimate
- Percentage above 1.0 (headroom)

---

## Requirements

| Package | Required | Purpose |
|---------|----------|---------|
| `openexr` | **Yes** | True 32-bit EXR writing |
| `imath` | **Yes** | OpenEXR dependency |
| `imageio` | Fallback | EXR writing if OpenEXR unavailable |
| `opencv-python` | Optional | Debanding filter |

---

## License

MIT

---

## Links

- GitHub: https://github.com/Jorge-Studio/js-exr-upbitrate
- ComfyUI: https://github.com/comfyanonymous/ComfyUI
