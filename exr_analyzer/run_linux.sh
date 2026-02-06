#!/bin/bash
# EXR Analyzer - Cinema VFX Diagnostic Tool
# Run this script on Linux

echo "============================================"
echo "  EXR Analyzer - Cinema VFX Diagnostic Tool"
echo "============================================"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed."
    echo "Please install Python 3.8+:"
    echo "  Ubuntu/Debian: sudo apt install python3 python3-venv python3-pip"
    echo "  Fedora: sudo dnf install python3 python3-pip"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to create virtual environment."
        echo "Try: sudo apt install python3-venv"
        exit 1
    fi
fi

# Activate virtual environment
source venv/bin/activate

# Install/update dependencies
echo "Checking dependencies..."
pip install -q -r requirements.txt
if [ $? -ne 0 ]; then
    echo "WARNING: Some dependencies may have failed to install."
    echo "For OpenEXR issues, try: sudo apt install libopenexr-dev"
    echo "Trying to continue anyway..."
fi

echo ""
echo "Starting EXR Analyzer..."
echo ""

# Suppress matplotlib config warnings
export MPLCONFIGDIR=/tmp/matplotlib

# Run the application
python exr_analyzer.py "$@"

# Deactivate on exit
deactivate
