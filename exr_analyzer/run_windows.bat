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
    echo.
    echo Please install Python 3.8+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

echo Python found:
python --version
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment.
        echo.
        pause
        exit /b 1
    )
    echo Virtual environment created.
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment.
    echo Trying to run without venv...
    goto :run_direct
)

REM Install/update dependencies
echo.
echo Installing dependencies...
pip install PyQt5 OpenEXR numpy matplotlib Pillow
if errorlevel 1 (
    echo.
    echo WARNING: Some dependencies may have failed to install.
    echo Trying to continue anyway...
    echo.
)

echo.
echo ============================================
echo   Starting EXR Analyzer...
echo ============================================
echo.

REM Run the application
python exr_analyzer.py %*
set EXITCODE=%errorlevel%

REM Deactivate on exit
deactivate

if %EXITCODE% neq 0 (
    echo.
    echo ============================================
    echo   Application exited with error code %EXITCODE%
    echo ============================================
    echo.
    pause
)

goto :eof

:run_direct
echo.
echo Installing dependencies globally...
pip install PyQt5 OpenEXR numpy matplotlib Pillow
echo.
echo Starting EXR Analyzer...
python exr_analyzer.py %*
if errorlevel 1 (
    echo.
    echo Application exited with an error.
    pause
)
