param(
    [int]$Hours = 720,
    [string]$Channel = "https://www.youtube.com/@sam_sulek/videos",
    [int]$PlaylistLimit = 50,
    [switch]$DryRun,
    [switch]$WaitForHandoff,
    [ValidateRange(1, 12)]
    [int]$TimeoutHours = 4
)

$ErrorActionPreference = "Stop"
$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$python = Join-Path $projectRoot "venv\Scripts\python.exe"
$worker = Join-Path $projectRoot "automation_worker.py"
$automationRoot = Join-Path $projectRoot "system\automation"
$stateFile = Join-Path $automationRoot "latest_state.json"
$lockFile = Join-Path $automationRoot "worker.lock.json"

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "VerbaCut virtual-environment Python is missing: $python"
}
if (-not (Test-Path -LiteralPath $worker -PathType Leaf)) {
    throw "VerbaCut automation worker is missing: $worker"
}

New-Item -ItemType Directory -Path $automationRoot -Force | Out-Null
$runId = $null
$process = $null
$attachedToExisting = $false

if (Test-Path -LiteralPath $lockFile -PathType Leaf) {
    try {
        $existingLock = Get-Content -LiteralPath $lockFile -Raw | ConvertFrom-Json
        $existingProcess = Get-Process -Id ([int]$existingLock.pid) -ErrorAction SilentlyContinue
        if ($existingProcess) {
            $runId = [string]$existingLock.run_id
            $process = $existingProcess
            $attachedToExisting = $true
        }
    }
    catch {
        # The worker owns stale-lock recovery; continue to a normal launch.
    }
}

if (-not $attachedToExisting) {
    $runId = Get-Date -Format "yyyyMMdd-HHmmss"
}
$arguments = @(
    $worker,
    "--run-id", $runId,
    "--hours", $Hours,
    "--channel", $Channel,
    "--playlist-limit", $PlaylistLimit
)
if ($DryRun -and -not $attachedToExisting) {
    $arguments += "--dry-run"
}

if (-not $attachedToExisting) {
    $process = Start-Process -FilePath $python `
        -ArgumentList $arguments `
        -WorkingDirectory $projectRoot `
        -WindowStyle Hidden `
        -PassThru
}

$deadline = (Get-Date).AddSeconds(60)
$acceptedStages = @("starting", "fetching", "links_fetched", "engine_started", "engine_running", "downloads_complete", "complete", "dry_run_complete")
do {
    Start-Sleep -Milliseconds 500
    if (Test-Path -LiteralPath $stateFile -PathType Leaf) {
        try {
            $state = Get-Content -LiteralPath $stateFile -Raw | ConvertFrom-Json
            if ($state.run_id -eq $runId -and $acceptedStages -contains $state.stage) {
                $accepted = [pscustomobject]@{
                    accepted = $true
                    run_id = $runId
                    worker_pid = [int]$state.pid
                    stage = $state.stage
                    state_file = $stateFile
                    worker_log = (Join-Path $automationRoot "logs\$runId.log")
                }
                if (-not $WaitForHandoff) {
                    if ($attachedToExisting) {
                        $accepted | Add-Member -NotePropertyName status -NotePropertyValue "already_running"
                    }
                    $accepted | ConvertTo-Json -Compress
                    exit 0
                }
                break
            }
            if ($state.run_id -eq $runId -and $state.stage -eq "failed") {
                throw "VerbaCut automation failed to start: $($state.error)"
            }
        }
        catch [System.Management.Automation.PSInvalidCastException] {
            # The worker writes atomically; retry a transient read during replacement.
        }
    }
    if ($process.HasExited) {
        throw "VerbaCut automation exited before startup was confirmed (exit $($process.ExitCode)). See $stateFile"
    }
} while ((Get-Date) -lt $deadline)

if (-not $accepted) {
    throw "Timed out waiting for VerbaCut automation startup confirmation. Worker pid: $($process.Id)"
}

$handoffDeadline = (Get-Date).AddHours($TimeoutHours)
$postDownloadSteps = @(
    "transcribing", "chunking", "ai_scanning", "merging_segments",
    "filtering_candidates", "extracting_clips", "completed"
)
do {
    Start-Sleep -Seconds 30
    if (-not (Test-Path -LiteralPath $stateFile -PathType Leaf)) {
        continue
    }
    try {
        $state = Get-Content -LiteralPath $stateFile -Raw | ConvertFrom-Json
    }
    catch {
        continue
    }
    if ($state.run_id -ne $runId) {
        throw "The VerbaCut state file was replaced by another run before handoff."
    }
    if ($state.stage -eq "failed" -or $state.status -in @("failed", "download_failed", "engine_stopped_early")) {
        throw "VerbaCut failed before transcription handoff: $($state.error) $($state.status)"
    }
    if ($state.status -eq "no_new_videos") {
        [pscustomobject]@{
            accepted = $true
            run_id = $runId
            worker_pid = [int]$state.pid
            status = "no_new_videos"
            state_file = $stateFile
        } | ConvertTo-Json -Compress
        exit 0
    }
    if ($state.stage -eq "dry_run_complete") {
        [pscustomobject]@{
            accepted = $true
            run_id = $runId
            worker_pid = [int]$state.pid
            status = "dry_run_complete"
            state_file = $stateFile
        } | ConvertTo-Json -Compress
        exit 0
    }
    if (
        $state.download_handoff_complete -eq $true -and
        $postDownloadSteps -contains $state.engine_step
    ) {
        [pscustomobject]@{
            accepted = $true
            run_id = $runId
            worker_pid = [int]$state.pid
            status = "handed_off_to_offline_engine"
            engine_step = $state.engine_step
            downloaded = [int]$state.download_summary.downloaded
            state_file = $stateFile
            worker_continues_offline = ($state.engine_step -ne "completed")
        } | ConvertTo-Json -Compress
        exit 0
    }
} while ((Get-Date) -lt $handoffDeadline)

throw "Timed out waiting for all downloads and transcription handoff after $TimeoutHours hour(s). See $stateFile"
