@echo off
REM Check if Python is installed
python --version > nul 2>&1
if %errorlevel% equ 0 (
    echo Python is installed.
    echo Running run.py...
    python run.py
) else (
    echo Python is not installed on this system.
    echo Please install Python to run this script.
)
pause