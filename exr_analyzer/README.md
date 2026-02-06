# EXR Analyzer — Cinema VFX Pipeline Diagnostic Tool

A professional GUI application for analyzing EXR files, measuring bit depth quality, and visualizing waveforms/histograms.

![EXR Analyzer Screenshot](screenshot.png)

## Features

- **Bit Depth Analysis**: Counts unique values to determine effective bit depth
- **Quality Rating**: Cinema-grade (5★) to 8-bit equivalent (1★) ratings
- **Waveform Display**: DaVinci Resolve-style RGB waveform visualization
- **Histogram**: RGB histogram with zoom/pan
- **Image Preview**: Tone-mapped HDR preview with exposure control
- **File Comparison**: Compare two EXR files side-by-side
- **Color Space Detection**: Identifies ACES, Rec.709, Rec.2020, DCI-P3
- **Encoding Detection**: Detects Linear vs Log encoding

## Quality Metrics

| Rating | Effective Bits | Unique Values | Use Case |
|--------|---------------|---------------|----------|
| ★★★★★ Cinema-grade | 13+ bits | 8,000+ | Professional VFX |
| ★★★★☆ Good | 11.5-13 bits | 3,000-8,000 | High-end production |
| ★★★☆☆ Acceptable | 10-11.5 bits | 1,000-3,000 | Standard production |
| ★★☆☆☆ Poor | 8.5-10 bits | 360-1,000 | Limited grading |
| ★☆☆☆☆ 8-bit equivalent | <8.5 bits | <360 | Not recommended |

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Windows

1. **Quick Start (Recommended)**
   ```
   Double-click: run_windows.bat
   ```

2. **Manual Installation**
   ```cmd
   cd exr_analyzer
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   python exr_analyzer.py
   ```

### macOS / Linux

1. **Quick Start (Recommended)**
   ```bash
   chmod +x run_mac.command
   ./run_mac.command
   ```
   Or double-click `run_mac.command` in Finder.

2. **Manual Installation**
   ```bash
   cd exr_analyzer
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   python exr_analyzer.py
   ```

---

## Usage

### Launch the Application

```bash
# No arguments - opens file picker
python exr_analyzer.py

# With file argument - opens directly
python exr_analyzer.py /path/to/your/file.exr
```

### Analyzing Files

1. Click **"Open EXR"** to select a file
2. View the quality metrics on the right panel
3. Use scroll wheel to zoom, right-drag to pan on visualizations
4. Click the **⛶** button to view any panel fullscreen

### Comparing Files

1. Open the first EXR file
2. Click **"Compare"** to open a second file
3. View side-by-side comparison in the Comparison panel

### Understanding the Results

- **Range**: The min-max values in the file (0.0-1.0 is SDR, above 1.0 is HDR)
- **Above 1.0**: Percentage of pixels with HDR values
- **Unique Values**: Number of distinct values (higher = better quality)
- **Midtone Step**: How much finer than 8-bit (higher = better)
- **Effective Bits**: Estimated bit depth based on unique values

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Scroll | Zoom in/out |
| Right-drag | Pan |
| Esc | Close fullscreen |

---

## Troubleshooting

### Windows: "OpenEXR not found"
Install the OpenEXR library:
```cmd
pip install OpenEXR
```

If that fails, you may need to install from a wheel:
```cmd
pip install --upgrade pip
pip install OpenEXR
```

### macOS: "No module named PyQt5"
```bash
pip install PyQt5
```

### Linux: Missing system dependencies
```bash
# Ubuntu/Debian
sudo apt-get install python3-pyqt5 libopenexr-dev

# Fedora
sudo dnf install python3-qt5 openexr-devel
```

### "Fontconfig error: No writable cache directories"
This is a harmless warning on some systems. The app will still work.

---

## License

MIT License - Free for commercial and personal use.

---

## Credits

Built for the js-exr-upbitrate ComfyUI node package.
