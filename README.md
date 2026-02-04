# High Bit Depth Export (ks-cn-high-bit-export)

ComfyUI custom nodes for **high bit-depth image and video export**: 16/32-bit EXR and **4K ProRes 4444 XQ** with optional linear color and debanding. Similar in spirit to [Luminance-Stack-Processor](https://github.com/sumitchatterjee13/Luminance-Stack-Processor) — preserves dynamic range and supports professional mastering pipelines.

## Where to find the nodes

Right‑click on the canvas → **Add Node**:

| Node | Menu path |
|------|-----------|
| Prepare Image High Bit Depth | **image** → **processing** |
| Save Image EXR | **image** → **save** |
| Save Video EXR Sequence | **image** → **video** |
| Save Video ProRes HQ (4K) | **image** → **video** |

Or use the search box (e.g. type `EXR` or `ProRes` or `High Bit`).

## Test workflow

A minimal test workflow is in **`workflows/test_high_bit_image.json`**:  
Empty Image → Prepare Image High Bit Depth → Save Image EXR. Load it via **Load** (or drag‑and‑drop), then **Queue Prompt**. Output goes to your ComfyUI output folder as a 32‑bit EXR (requires `imageio`; for EXR support see Requirements below).

## Nodes

| Node | Description |
|------|-------------|
| **Prepare Image High Bit Depth** | Optional sRGB→linear and bilateral debanding. Use before EXR/ProRes save. |
| **Save Image EXR** | Save a single image as 16- or 32-bit EXR (zip/rle/piz compression). |
| **Save Video EXR Sequence** | Save video as an EXR sequence (16/32 bit). Use with ffmpeg for ProRes later if needed. |
| **Save Video ProRes HQ (4K)** | Save video as ProRes 4444 XQ (12-bit) with optional **linear → gradfun deband → bt709** pipeline. |

## 4K ProRes pipeline (when ffmpeg is available)

**Save Video ProRes HQ** uses the same idea as the “simplest good” 4K ProRes mastering approach:

- **Option 1 (default):** ProRes 4444 XQ mezzanine  
  - Internally: float → linear → `gradfun=20:16` (deband) → bt709 → `yuv444p10le` → `prores_ks` profile 5.
- **Option 2:** Use **Save Video EXR Sequence** for a true 16/32-bit EXR sequence, then run ffmpeg yourself to encode ProRes from the EXRs.

If **ffmpeg** is not in `PATH`, the node falls back to PyAV-only encoding (no gradfun/zscale).

### Example: ProRes from EXR sequence (manual)

```bash
ffmpeg -framerate 24 -i "output/video/ComfyUI_EXR_00001_%06d.exr" \
  -vf "format=gbrpf32le,zscale=t=linear,gradfun=20:16,zscale=t=bt709,format=yuv444p10le" \
  -c:v prores_ks -profile:v 5 -pix_fmt yuv444p10le \
  -c:a pcm_s16le output_prores4444xq.mov
```

## Requirements

- **opencv-python** ≥ 4.8 — **recommended for Save Image EXR**: used as the primary EXR writer (reliable on Windows). Also used for **Prepare Image High Bit Depth** debanding (bilateral filter).
- **imageio** ≥ 2.31 — required for EXR when OpenCV is not available, and for Save Video EXR Sequence / ProRes HQ (temp EXR frames when using ffmpeg).
- **ffmpeg** in `PATH` — optional; enables the full ProRes pipeline (gradfun + zscale) in **Save Video ProRes HQ**. PyAV (usually bundled with ComfyUI) is used when ffmpeg is not available.

**Save Image EXR:** Uses **imageio with FreeImage** (install `imageio[freeimage]`). On first EXR save, imageio may download the FreeImage DLL (~7 MB) once. Standard opencv-python from pip has EXR disabled, so EXR writing uses imageio’s FreeImage plugin.

### Install (portable)

From your ComfyUI **portable root**:

```bat
python_embeded\python.exe -m pip install imageio opencv-python
```

Or:

```bat
python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ks-cn-high-bit-export\requirements.txt
```

## Usage

1. **Image:** VAE Decode → (optional) **Prepare Image High Bit Depth** (linear + deband) → **Save Image EXR** (16 or 32 bit).
2. **Video:** Decode video as usual → (optional) prepare frames for HDR → **Save Video ProRes HQ** for 4K ProRes 4444 XQ, or **Save Video EXR Sequence** for EXR frames and then ffmpeg to ProRes.

## Verifying output bit depth

The save nodes write 16- or 32-bit EXR according to the **bit_depth** setting. To check that files are stored in the requested depth:

- **OpenCV (EXR enabled):** 16-bit uses `IMWRITE_EXR_TYPE_HALF`, 32-bit uses `IMWRITE_EXR_TYPE_FLOAT`.
- **imageio/FreeImage:** 16-bit is written as half; 32-bit may still be stored as half depending on the plugin.

A small test script reads the EXR header and asserts the channel type (HALF = 16, FLOAT = 32). Run from the ComfyUI portable root:

```bat
python_embeded\python.exe ComfyUI\custom_nodes\ks-cn-high-bit-export\tests\test_exr_bit_depth.py
```

If no EXR backend is available (OpenCV EXR disabled and no imageio FreeImage), the tests are skipped. With `imageio[freeimage]` installed, the 32-bit and size-comparison tests run; the 16-bit test runs when OpenCV has EXR support or imageio can write EXR.

## License

MIT.
