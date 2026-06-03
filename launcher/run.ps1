$ErrorActionPreference = "Stop"

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $projectRoot
$venvPython = Join-Path $projectRoot "venv\Scripts\python.exe"
$commandInstaller = Join-Path $PSScriptRoot "install-command.ps1"

if (Test-Path $commandInstaller) {
    try {
        & $commandInstaller -Quiet
    }
    catch {
        Write-Host "Warning: Could not install the 'aiclip' terminal shortcut automatically." -ForegroundColor Yellow
    }
}

try {
    python --version | Out-Null
}
catch {
    Write-Host "Error: Python is not installed or not in PATH." -ForegroundColor Red
    Write-Host "Install Python 3.10+ from https://www.python.org/downloads/"
    exit 1
}

if (-not (Test-Path "venv\Scripts\python.exe")) {
    Write-Host "Creating virtual environment..."
    python -m venv venv
}

Write-Host "Preparing runtime dependencies..."
& $venvPython setup_env.py --torch auto
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Dependency setup failed." -ForegroundColor Red
    exit $LASTEXITCODE
}

& $venvPython main.py @args
exit $LASTEXITCODE

