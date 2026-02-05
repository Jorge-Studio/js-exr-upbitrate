# High Bit Depth EXR Export for ComfyUI (v2.0)

Professional-grade EXR export with **Log format support**, **color grading controls**, and **maximum tonal precision** for VFX, compositing, and color grading workflows.

---

## What's New in v2.0

- **Log Format Export**: ARRI LogC3, Sony S-Log3, Panasonic V-Log, Canon Log 3, RED Log3G10, DaVinci Intermediate
- **Color Space Converter**: Convert between sRGB, Linear, and Log formats
- **Color Grading Controller**: Professional lift/gamma/gain, exposure, contrast, saturation
- **HDR Curve Editor**: Lightroom-style shadows/midtones/highlights controls
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

## Nodes Included

| Node | Description |
|------|-------------|
| **Color Space Converter** | Convert between sRGB, Linear, and Log formats |
| **Color Grading Controller** | Exposure, contrast, lift/gamma/gain, saturation |
| **HDR Curve Editor** | Shadows, midtones, highlights, whites, blacks |
| **Prepare Image High Bit Depth** | sRGB→Linear + headroom + debanding |
| **Save Image EXR** | Export single image as 16/32-bit EXR with Log format |
| **Save Video EXR Sequence** | Export video as EXR sequence with Log format |
| **Image Stats** | Display range, unique values, bit depth estimate |

---

## Recommended Workflow for Professional Output

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
