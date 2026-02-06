#!/bin/bash
# EXR Analyzer - macOS Launcher

echo "============================================"
echo "  EXR Analyzer - Cinema VFX Diagnostic Tool"
echo "============================================"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed."
    echo "Please install Python from https://www.python.org/downloads/"
    echo ""
    read -p "Press Enter to exit..."
    exit 1
fi

echo "Python found: $(python3 --version)"
echo ""

# Create virtual environment if needed
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to create virtual environment."
        read -p "Press Enter to exit..."
        exit 1
    fi
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo ""
echo "Checking dependencies..."
pip install -q PyQt5 OpenEXR numpy matplotlib Pillow

echo ""
echo "============================================"
echo "  Starting EXR Analyzer..."
echo "============================================"
echo ""

# Run the application
python exr_analyzer.py "$@"
EXIT_CODE=$?

# Deactivate
deactivate

if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "Application exited with error code $EXIT_CODE"
    read -p "Press Enter to exit..."
fi
