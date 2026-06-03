@echo off
setlocal
cd /d "%~dp0\.."

powershell -NoProfile -ExecutionPolicy Bypass -File "launcher\install-command.ps1" -Quiet >nul 2>&1
if errorlevel 1 (
    echo Warning: Could not install the 'verbacut' terminal shortcut automatically.
)

python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH.
    echo Install Python 3.10+ from https://www.python.org/downloads/
    exit /b 1
)

if not exist "venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo Error: Failed to create virtual environment.
        exit /b 1
    )
)

echo Preparing runtime dependencies...
venv\Scripts\python.exe setup_env.py --torch auto
if errorlevel 1 (
    echo Error: Dependency setup failed.
    exit /b 1
)

venv\Scripts\python.exe main.py %*
set "EXIT_CODE=%ERRORLEVEL%"

exit /b %EXIT_CODE%

