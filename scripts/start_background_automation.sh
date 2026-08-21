#!/usr/bin/env bash
# Start or attach to a VerbaCut intake and exit after its offline handoff.
set -euo pipefail

hours=720
channel="https://www.youtube.com/@sam_sulek/videos"
playlist_limit=50
dry_run=false
wait_for_handoff=false
timeout_hours=4

usage() {
    cat <<'EOF'
Usage: start_background_automation.sh [options]
  --hours N
  --channel URL
  --playlist-limit N
  --dry-run
  --wait-for-handoff
  --timeout-hours N
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --hours) hours="$2"; shift 2 ;;
        --channel) channel="$2"; shift 2 ;;
        --playlist-limit) playlist_limit="$2"; shift 2 ;;
        --dry-run) dry_run=true; shift ;;
        --wait-for-handoff) wait_for_handoff=true; shift ;;
        --timeout-hours) timeout_hours="$2"; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

for value in "$hours" "$playlist_limit" "$timeout_hours"; do
    if [[ ! "$value" =~ ^[0-9]+$ ]]; then
        printf 'Numeric options must be non-negative integers.\n' >&2
        exit 2
    fi
done
if ((hours < 1 || hours > 8760 || playlist_limit < 1 || playlist_limit > 200 || timeout_hours < 1 || timeout_hours > 12)); then
    printf 'Options are outside the supported range.\n' >&2
    exit 2
fi

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python="$project_root/.venv/bin/python"
worker="$project_root/automation_worker.py"
automation_root="$project_root/system/automation"
state_file="$automation_root/latest_state.json"
lock_file="$automation_root/worker.lock.json"
runs_root="$automation_root/runs"

if [[ ! -x "$python" ]]; then
    printf 'VerbaCut virtual-environment Python is missing: %s\n' "$python" >&2
    exit 1
fi
if [[ ! -f "$worker" ]]; then
    printf 'VerbaCut automation worker is missing: %s\n' "$worker" >&2
    exit 1
fi
if ! command -v setsid >/dev/null 2>&1; then
    printf 'setsid is required to detach the offline VerbaCut worker.\n' >&2
    exit 1
fi

mkdir -p "$automation_root" "$runs_root" "$automation_root/logs"

state_snapshot() {
    "$python" - "$state_file" <<'PY'
import json
import sys

try:
    with open(sys.argv[1], encoding="utf-8") as handle:
        state = json.load(handle)
except (OSError, ValueError):
    raise SystemExit(1)

print("\t".join(str(state.get(key, "")) for key in (
    "run_id", "pid", "stage", "status", "engine_step", "download_handoff_complete", "error"
)))
PY
}

emit_result() {
    "$python" - "$@" <<'PY'
import json
import sys

payload = {
    "accepted": True,
    "run_id": sys.argv[1],
    "worker_pid": int(sys.argv[2]),
    "status": sys.argv[3],
    "state_file": sys.argv[4],
}
if len(sys.argv) > 5 and sys.argv[5]:
    payload["engine_step"] = sys.argv[5]
if len(sys.argv) > 6:
    payload["worker_continues_offline"] = sys.argv[6] == "true"
print(json.dumps(payload, separators=(",", ":")))
PY
}

run_id=""
worker_pid=""
attached=false
resume=false

if [[ -f "$lock_file" ]]; then
    lock_values="$($python - "$lock_file" <<'PY' 2>/dev/null || true
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    lock = json.load(handle)
print(f"{int(lock.get('pid') or 0)}\t{lock.get('run_id') or ''}")
PY
)"
    if [[ -n "$lock_values" ]]; then
        IFS=$'\t' read -r existing_pid existing_run_id <<<"$lock_values"
        if [[ "$existing_pid" =~ ^[1-9][0-9]*$ ]] && kill -0 "$existing_pid" 2>/dev/null; then
            run_id="$existing_run_id"
            worker_pid="$existing_pid"
            attached=true
        fi
    fi
fi

if [[ "$attached" == false && -f "$state_file" ]]; then
    resume_id="$($python - "$state_file" "$runs_root" <<'PY' 2>/dev/null || true
import json
import sys
from pathlib import Path

try:
    with open(sys.argv[1], encoding="utf-8") as handle:
        state = json.load(handle)
    runs_root = Path(sys.argv[2]).resolve()
    run_dir = Path(str(state.get("run_dir") or "")).resolve()
    valid_dir = run_dir.is_relative_to(runs_root)
    has_input = any(path.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"} for path in (run_dir / "input").glob("*"))
    unfinished = state.get("stage") in {"engine_started", "engine_running", "downloads_complete", "failed"}
    if valid_dir and has_input and unfinished and state.get("run_id"):
        print(state["run_id"])
except (OSError, ValueError):
    pass
PY
)"
    if [[ -n "$resume_id" ]]; then
        run_id="$resume_id"
        resume=true
    fi
fi

if [[ -z "$run_id" ]]; then
    run_id="$(date +%Y%m%d-%H%M%S)"
fi

if [[ "$attached" == false ]]; then
    worker_args=("$worker" --run-id "$run_id" --hours "$hours" --channel "$channel" --playlist-limit "$playlist_limit")
    [[ "$dry_run" == true ]] && worker_args+=(--dry-run)
    [[ "$resume" == true ]] && worker_args+=(--resume-run)
    setsid "$python" "${worker_args[@]}" </dev/null >/dev/null 2>&1 &
    worker_pid=$!
fi

deadline=$((SECONDS + 60))
accepted_stages=(starting fetching links_fetched engine_started engine_running downloads_complete complete dry_run_complete)
while ((SECONDS < deadline)); do
    sleep 0.5
    snapshot="$(state_snapshot 2>/dev/null || true)"
    if [[ -n "$snapshot" ]]; then
        IFS=$'\t' read -r state_run_id state_pid stage status engine_step handoff error <<<"$snapshot"
        if [[ "$state_run_id" == "$run_id" ]]; then
            if [[ "$stage" == "failed" ]]; then
                printf 'VerbaCut automation failed to start: %s\n' "$error" >&2
                exit 1
            fi
            for accepted_stage in "${accepted_stages[@]}"; do
                if [[ "$stage" == "$accepted_stage" ]]; then
                    worker_pid="$state_pid"
                    if [[ "$wait_for_handoff" == false ]]; then
                        start_status="started"
                        [[ "$attached" == true ]] && start_status="already_running"
                        emit_result "$run_id" "$worker_pid" "$start_status" "$state_file" "" true
                        exit 0
                    fi
                    break 2
                fi
            done
        fi
    fi
    if [[ "$attached" == false ]] && ! kill -0 "$worker_pid" 2>/dev/null; then
        printf 'VerbaCut automation exited before startup was confirmed. See %s\n' "$state_file" >&2
        exit 1
    fi
done

if [[ "${stage:-}" != "starting" && "${stage:-}" != "fetching" && "${stage:-}" != "links_fetched" && "${stage:-}" != "engine_started" && "${stage:-}" != "engine_running" && "${stage:-}" != "downloads_complete" && "${stage:-}" != "complete" && "${stage:-}" != "dry_run_complete" ]]; then
    printf 'Timed out waiting for VerbaCut startup confirmation. See %s\n' "$state_file" >&2
    exit 1
fi

handoff_deadline=$((SECONDS + timeout_hours * 3600))
post_download_steps=(transcribing chunking ai_scanning merging_segments filtering_candidates extracting_clips completed)
while ((SECONDS < handoff_deadline)); do
    sleep 30
    snapshot="$(state_snapshot 2>/dev/null || true)"
    [[ -z "$snapshot" ]] && continue
    IFS=$'\t' read -r state_run_id state_pid stage status engine_step handoff error <<<"$snapshot"
    if [[ "$state_run_id" != "$run_id" ]]; then
        printf 'The VerbaCut state file was replaced by another run before handoff.\n' >&2
        exit 1
    fi
    if [[ "$stage" == "failed" || "$status" == "failed" || "$status" == "download_failed" || "$status" == "engine_stopped_early" ]]; then
        printf 'VerbaCut failed before transcription handoff: %s\n' "$error" >&2
        exit 1
    fi
    if [[ "$status" == "no_new_videos" ]]; then
        emit_result "$run_id" "$state_pid" no_new_videos "$state_file" "" false
        exit 0
    fi
    if [[ "$stage" == "dry_run_complete" ]]; then
        emit_result "$run_id" "$state_pid" dry_run_complete "$state_file" "" false
        exit 0
    fi
    if [[ "$handoff" == "True" || "$handoff" == "true" ]]; then
        for post_download_step in "${post_download_steps[@]}"; do
            if [[ "$engine_step" == "$post_download_step" ]]; then
                continues=true
                [[ "$engine_step" == "completed" ]] && continues=false
                emit_result "$run_id" "$state_pid" handed_off_to_offline_engine "$state_file" "$engine_step" "$continues"
                exit 0
            fi
        done
    fi
done

printf 'Timed out waiting for all downloads and transcription handoff after %s hour(s). See %s\n' "$timeout_hours" "$state_file" >&2
exit 1
