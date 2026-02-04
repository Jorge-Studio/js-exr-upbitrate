# High Bit Depth EXR Export for ComfyUI

Export images and videos as **16-bit or 32-bit EXR files** for professional VFX, color grading, and compositing workflows.

---

## Installation

### Step 1: Clone the Repository

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Jorge-Studio/js-exr-upbitrate.git
```

### Step 2: Install Dependencies

**Option A: Standard Python**
```bash
pip install openexr imath imageio opencv-python
```

**Option B: ComfyUI Portable (Windows)**
```bat
python_embeded\python.exe -m pip install openexr imath imageio opencv-python
```

**Option C: From requirements.txt**
```bash
pip install -r ComfyUI/custom_nodes/js-exr-upbitrate/requirements.txt
```

### Step 3: Restart ComfyUI

Restart ComfyUI or press `Ctrl+Shift+R` to refresh.

---

## Nodes Included

| Node | What It Does |
|------|--------------|
| **Prepare Image High Bit Depth** | Optional preprocessing: sRGB→linear conversion and debanding |
| **Save Image EXR** | Save a single image as 16-bit or 32-bit EXR |
| **Save Video EXR Sequence** | Save video frames as a numbered EXR sequence |

### Finding the Nodes

Right-click canvas → **Add Node** → **image** → **processing** or **save**

Or double-click and search: `EXR`, `High Bit`, or `Prepare`

---

## Quick Start Guide

### Export a Single Image as 32-bit EXR

```
[Any Image Source] → Save Image EXR
                     └─ bit_depth: 32
                     └─ compression: zip
```

**Steps:**
1. Connect any image output to **Save Image EXR**
2. Set **bit_depth** to `32` for maximum quality
3. Set **filename_prefix** (e.g., `my_image`)
4. Queue the prompt
5. Find your EXR in `ComfyUI/output/`

---

### Export Video Frames as 32-bit EXR Sequence

```
Load Video (Upload) → Save Video EXR Sequence → Preview Image
                      └─ bit_depth: 32
                      └─ compression: zip
```

**Steps:**
1. Add **Load Video (Upload)** node (from VideoHelperSuite)
2. Upload your video file
3. Connect to **Save Video EXR Sequence**
4. Set **bit_depth** to `32`
5. Queue the prompt
6. Find your EXR sequence in `ComfyUI/output/`

Output files: `ComfyUI_EXR_seq_00001_000000.exr`, `_000001.exr`, `_000002.exr`, etc.

---

### Optional: Prepare Image for HDR/Linear

For best results in VFX pipelines, use **Prepare Image High Bit Depth** before saving:

```
[Image Source] → Prepare Image High Bit Depth → Save Image EXR
                 └─ output_linear: true (converts sRGB to linear)
                 └─ deband_strength: 0.5 (removes banding artifacts)
```

---

## Node Settings Explained

### Save Image EXR / Save Video EXR Sequence

| Setting | Options | Description |
|---------|---------|-------------|
| **bit_depth** | `16` or `32` | 16-bit (HALF) for smaller files, 32-bit (FLOAT) for maximum precision |
| **compression** | `zip`, `zips`, `piz`, `rle`, `none` | `zip` recommended for best size/speed balance |
| **filename_prefix** | any string | Base name for output files |

### Prepare Image High Bit Depth

| Setting | Default | Description |
|---------|---------|-------------|
| **output_linear** | `false` | Convert from sRGB to linear color space |
| **deband_strength** | `0.0` | Remove banding (0.0 = off, 0.5-1.0 = typical, 2.0 = maximum) |

---

## Verifying Your EXR Bit Depth

### On Mac (Preview.app)
1. Open the EXR file in Preview
2. Press `Cmd+I` to open Inspector
3. Check **Depth**: should show `16` or `32`
4. Check **Is Float**: should show `1` (floating point)

### On Windows/Linux (exiftool)
```bash
exiftool your_file.exr | grep -i "bit"
```

### Why Does My EXR Look Washed Out?

EXR files are saved in **linear color space** (correct for VFX). Preview apps don't apply gamma correction, so they look washed out.

**To view correctly, use:**
- DaVinci Resolve
- After Effects
- Nuke
- Photoshop (with proper color settings)

---

## Importing EXR into Professional Software

### DaVinci Resolve
1. Import EXR sequence via **Media Pool**
2. Right-click → **Clip Attributes** → Set frame rate
3. Color space is automatically detected as linear

### After Effects
1. Import → select first EXR of sequence
2. Check **OpenEXR Sequence**
3. Use **Color Profile Converter** if needed

### Converting EXR Sequence to ProRes (FFmpeg)

```bash
ffmpeg -framerate 24 -i "ComfyUI_EXR_seq_00001_%06d.exr" \
  -vf "format=gbrpf32le,zscale=t=linear,zscale=t=bt709,format=yuv444p10le" \
  -c:v prores_ks -profile:v 5 -pix_fmt yuv444p10le \
  output_prores4444xq.mov
```

---

## Requirements Summary

| Package | Required | Purpose |
|---------|----------|---------|
| `openexr` | **Yes** | True 32-bit EXR writing |
| `imath` | **Yes** | OpenEXR dependency |
| `imageio` | Fallback | EXR writing if OpenEXR unavailable |
| `opencv-python` | Optional | Debanding in Prepare node |

---

## Troubleshooting

### Node Not Appearing
1. Restart ComfyUI
2. Check terminal for import errors
3. Verify dependencies are installed

### 16-bit Instead of 32-bit
Make sure `openexr` and `imath` are installed:
```bash
pip install openexr imath
```

### "Could not write EXR" Error
Install the EXR backend:
```bash
pip install openexr imath imageio
```

---

## Example Workflows

Test workflows are in the `workflows/` folder:
- `test_high_bit_image.json` - Single image to EXR

---

## License

MIT

---

## Links

- GitHub: https://github.com/Jorge-Studio/js-exr-upbitrate
- ComfyUI: https://github.com/comfyanonymous/ComfyUI
