param(
    [string]$CommandName = "aiclip",
    [switch]$Quiet
)

$ErrorActionPreference = "Stop"

function Write-Status {
    param(
        [string]$Message,
        [string]$Color = "White"
    )

    if (-not $Quiet) {
        Write-Host $Message -ForegroundColor $Color
    }
}

if ($CommandName -ne "aiclip") {
    Write-Host "Only the built-in command name 'aiclip' is currently supported." -ForegroundColor Yellow
    exit 1
}

$launcherDir = [System.IO.Path]::GetFullPath($PSScriptRoot)
$commandPath = Join-Path $launcherDir "$CommandName.cmd"

if (-not (Test-Path $commandPath)) {
    Write-Host "Command wrapper not found: $commandPath" -ForegroundColor Red
    exit 1
}

$currentPath = [Environment]::GetEnvironmentVariable("Path", "User")
$pathParts = @()
if (-not [string]::IsNullOrWhiteSpace($currentPath)) {
    $pathParts = $currentPath -split ";" | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
}

$alreadyInstalled = $false
foreach ($part in $pathParts) {
    try {
        if ([System.IO.Path]::GetFullPath($part).TrimEnd("\") -ieq $launcherDir.TrimEnd("\")) {
            $alreadyInstalled = $true
            break
        }
    }
    catch {
        continue
    }
}

if (-not $alreadyInstalled) {
    $updatedParts = @($pathParts + $launcherDir)
    [Environment]::SetEnvironmentVariable("Path", ($updatedParts -join ";"), "User")
    Write-Status "Installed command: $CommandName" "Green"
    Write-Status "Open a new terminal, then run: $CommandName"
}
else {
    Write-Status "Command already installed: $CommandName" "Green"
}
