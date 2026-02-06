# EXR Analyzer - Cinema VFX Diagnostic Tool

A professional-grade GUI application for analyzing EXR files and diagnosing bit depth quality issues in VFX pipelines.

## Features

- **Bit Depth Analysis**: Measures effective bit depth by counting unique values per channel
- **Quality Rating**: Cinema-grade (20+ bits) to 8-bit equivalent ratings
- **Waveform Display**: DaVinci Resolve-style waveform with RGB overlay
- **Histogram**: Interactive RGB histogram with zoom/pan
- **Image Preview**: Tone-mapped preview with exposure control
- **File Comparison**: Compare two EXR files side-by-side
- **Color Space Detection**: Identifies ACES, Rec.709, Rec.2020, DCI-P3

## Quality Metrics

| Effective Bits | Rating | Meaning |
|----------------|--------|---------|
| 13+ bits | Cinema-grade | Full 16-bit half precision preserved |
| 11.5+ bits | Good | Suitable for most VFX work |
| 10+ bits | Acceptable | Some quantization visible |
| 8.5+ bits | Poor | Significant quality loss |
| <8.5 bits | 8-bit equivalent | Major precision issues |

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Windows
1. Double-click `run_windows.bat`
2. The script will automatically:
   - Create a virtual environment
   - Install all dependencies
   - Launch the application

### macOS
1. Open Terminal
2. Navigate to this folder: `cd path/to/exr_analyzer`
3. Make the script executable: `chmod +x run_mac.command`
4. Double-click `run_mac.command` or run: `./run_mac.command`

### Linux
1. Open Terminal
2. Navigate to this folder: `cd path/to/exr_analyzer`
3. Make the script executable: `chmod +x run_linux.sh`
4. Run: `./run_linux.sh`

### Manual Installation
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run
python exr_analyzer.py
```

## Usage

### Opening Files
- Click "Open EXR" to select a file
- Or drag and drop an EXR file onto the window
- Or pass a file path as command line argument: `python exr_analyzer.py file.exr`

### Navigation
- **Scroll wheel**: Zoom in/out on any visualization
- **Right-click drag**: Pan the view
- **Reset button (⟲)**: Reset to default view
- **Fullscreen button (⛶)**: Open visualization in fullscreen

### Comparison Mode
1. Open a primary EXR file
2. Click "Compare" to load a second file
3. View side-by-side metrics

## Troubleshooting

### "Could not write EXR" error
Install OpenEXR: `pip install openexr imath`

### Application closes immediately (Windows)
The updated batch file will now pause and show any error messages. Common issues:
- Python not in PATH: Reinstall Python and check "Add to PATH"
- Missing Visual C++ Redistributable: Install from Microsoft

### PyQt5 import error
```bash
pip uninstall PyQt5 PyQt5-Qt5 PyQt5-sip
pip install PyQt5
```

### OpenEXR won't install
On Linux, you may need: `sudo apt install libopenexr-dev`
On macOS: `brew install openexr`

## Dependencies

- PyQt5 - GUI framework
- OpenEXR - EXR file reading
- numpy - Numerical analysis
- matplotlib - Visualization
- Pillow - Image handling

## License

Part of the js-exr-upbitrate ComfyUI custom node package.
