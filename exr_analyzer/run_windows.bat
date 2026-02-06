@echo off
title EXR Analyzer - Cinema VFX Diagnostic Tool
cd /d "%~dp0"

echo ============================================
echo   EXR Analyzer - Cinema VFX Diagnostic Tool
echo ============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH.
    echo Please install Python 3.8+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install/update dependencies
echo Checking dependencies...
pip install -q -r requirements.txt
if errorlevel 1 (
    echo WARNING: Some dependencies may have failed to install.
    echo Trying to continue anyway...
)

echo.
echo Starting EXR Analyzer...
echo.

REM Run the application
python exr_analyzer.py %*

REM Deactivate on exit
deactivate
