from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shutil
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import yt_dlp.version
from rich.console import Console

from core.clipping import scan_input_videos
from core.engine import ClippingEngine
from core.yt_handler import YTHandler
from utils.logging_setup import setup_logging
from utils.validators import ensure_runtime_layout, load_json_file, load_validated_config, save_json_file


MINIMUM_YT_DLP_VERSION = (2026, 7, 4)
AUTOMATION_PROFILE = Path("profiles") / "beyond_comfort_motivational.json"


def _version_key(value: str) -> tuple[int, int, int]:
    parts = [int(part) for part in re.findall(r"\d+", str(value))[:3]]
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        process = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
        if not process:
            return False
        ctypes.windll.kernel32.CloseHandle(process)
        return True
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _automation_paths(base_dir: Path) -> Dict[str, Path]:
    root = base_dir / "system" / "automation"
    return {
        "root": root,
        "lock": root / "worker.lock.json",
        "state": root / "latest_state.json",
        "runs": root / "runs",
        "logs": root / "logs",
    }


def _write_state(path: Path, payload: Dict[str, Any]) -> None:
    payload = dict(payload)
    payload["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    save_json_file(path, payload)


@contextmanager
def _worker_lock(path: Path, run_id: str) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            existing = load_json_file(path)
        except Exception:
            existing = {}
        existing_pid = int(existing.get("pid") or 0) if isinstance(existing, dict) else 0
        if _pid_is_running(existing_pid):
            raise RuntimeError(
                f"VerbaCut automation is already running (pid={existing_pid}, "
                f"run_id={existing.get('run_id', 'unknown')})."
            )
        path.unlink(missing_ok=True)

    handle = path.open("x", encoding="utf-8")
    try:
        json.dump({"pid": os.getpid(), "run_id": run_id}, handle, indent=2)
        handle.write("\n")
        handle.flush()
        handle.close()
        yield
    finally:
        if not handle.closed:
            handle.close()
        try:
            current = load_json_file(path)
        except Exception:
            current = {}
        if isinstance(current, dict) and int(current.get("pid") or 0) == os.getpid():
            path.unlink(missing_ok=True)


@contextmanager
def _plain_console(path: Path) -> Iterator[Console]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        yield Console(
            file=stream,
            force_terminal=False,
            color_system=None,
            no_color=True,
            width=120,
        )


class AutomationEngine(ClippingEngine):
    def __init__(self, *args: Any, state_file: Path, run_state: Dict[str, Any], **kwargs: Any) -> None:
        self._automation_state_file = state_file
        self._automation_run_state = run_state
        super().__init__(*args, **kwargs)

    def _publish(self, **updates: Any) -> None:
        self._automation_run_state.update(updates)
        _write_state(self._automation_state_file, self._automation_run_state)

    def _write_status(self, payload: Dict[str, Any]) -> None:
        super()._write_status(payload)
        step = str(payload.get("current_step") or "unknown")
        self._publish(stage="engine_running", engine_step=step)

    def _after_downloads(self, summary: Dict[str, Any]) -> None:
        downloaded_files = [str(path.resolve()) for path in scan_input_videos(self.input_dir)]
        successful = int(summary.get("downloaded", 0)) + int(
            summary.get("already_downloaded", 0)
        )
        total = int(summary.get("total", 0))
        self._publish(
            stage="downloads_complete",
            engine_step="downloads_complete",
            download_summary=summary,
            downloaded_files=downloaded_files,
            download_handoff_complete=(total > 0 and successful == total),
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch recent channel uploads and run VerbaCut unattended in the background."
    )
    parser.add_argument("--run-id")
    parser.add_argument("--hours", type=int)
    parser.add_argument("--channel")
    parser.add_argument("--playlist-limit", type=int)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-after-downloads", action="store_true")
    return parser


def _load_profile(base_dir: Path, requested: Optional[Path]) -> Dict[str, Any]:
    profile_path = (requested or (base_dir / AUTOMATION_PROFILE)).resolve()
    payload = load_json_file(profile_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Automation profile must be a JSON object: {profile_path}")
    for key in ("channel", "hours_limit", "playlist_limit", "user_query", "system_prompt"):
        if not payload.get(key):
            raise ValueError(f"Automation profile is missing required field: {key}")
    payload["profile_path"] = str(profile_path)
    return payload


def _preflight_errors(base_dir: Path, config: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    for executable in ("ffmpeg", "ffprobe", "ollama"):
        if shutil.which(executable) is None:
            errors.append(f"Required executable is not available on PATH: {executable}")
    model = str(config.get("ollama", {}).get("model", "")).strip()
    if not model:
        errors.append("Configured Ollama model is empty.")
    elif shutil.which("ollama") is not None:
        import subprocess

        probe = subprocess.run(
            ["ollama", "show", model],
            cwd=base_dir,
            capture_output=True,
            text=True,
            errors="replace",
            check=False,
            timeout=30,
        )
        if probe.returncode != 0:
            errors.append(f"Configured Ollama model is not installed locally: {model}")
    return errors


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    base_dir = Path(__file__).resolve().parent
    ensure_runtime_layout(base_dir)
    paths = _automation_paths(base_dir)
    for key in ("root", "runs", "logs"):
        paths[key].mkdir(parents=True, exist_ok=True)

    run_id = args.run_id or datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    run_dir = paths["runs"] / run_id
    run_state: Dict[str, Any] = {
        "run_id": run_id,
        "pid": os.getpid(),
        "stage": "starting",
        "base_dir": str(base_dir),
        "run_dir": str(run_dir),
    }

    try:
        with _worker_lock(paths["lock"], run_id):
            _write_state(paths["state"], run_state)
            profile = _load_profile(base_dir, args.profile)
            hours = int(args.hours if args.hours is not None else profile["hours_limit"])
            channel = str(args.channel or profile["channel"])
            playlist_limit = int(
                args.playlist_limit if args.playlist_limit is not None else profile["playlist_limit"]
            )
            if hours < 1 or hours > 8760:
                raise ValueError("--hours must be between 1 and 8760")
            if playlist_limit < 1 or playlist_limit > 200:
                raise ValueError("--playlist-limit must be between 1 and 200")

            installed_version = str(yt_dlp.version.__version__)
            if _version_key(installed_version) < MINIMUM_YT_DLP_VERSION:
                raise RuntimeError(
                    f"yt-dlp {installed_version} is below required version "
                    f"{'.'.join(str(part) for part in MINIMUM_YT_DLP_VERSION)}."
                )

            valid, loaded, errors = load_validated_config(base_dir)
            if not valid or loaded is None:
                raise RuntimeError("Invalid VerbaCut config; existing setup was preserved: " + "; ".join(errors))
            dependency_errors = _preflight_errors(base_dir, loaded)
            if dependency_errors:
                raise RuntimeError("Automation preflight failed: " + "; ".join(dependency_errors))

            logger = setup_logging(
                base_dir=base_dir,
                log_level=str(loaded["app"].get("log_level", "INFO")),
                dev_mode=bool(loaded["app"].get("dev_mode", False)),
                log_file=paths["logs"] / f"{run_id}.log",
            )
            handler = YTHandler(base_dir=base_dir, logger=logger)
            run_state.update(
                stage="fetching",
                channel=channel,
                hours_limit=hours,
                playlist_limit=playlist_limit,
                profile_path=profile["profile_path"],
                user_query=str(profile["user_query"]),
                yt_dlp_version=installed_version,
            )
            _write_state(paths["state"], run_state)

            links = handler.fetch_recent_links(
                channels=[channel],
                hours_limit=hours,
                playlistend=playlist_limit,
                mark_fetched=False,
                respect_fetched_history=False,
            )
            run_state.update(stage="links_fetched", link_count=len(links), links=links)
            _write_state(paths["state"], run_state)

            if args.dry_run:
                run_state.update(stage="dry_run_complete", status="ready", preflight="passed")
                _write_state(paths["state"], run_state)
                return 0
            if not links:
                run_state.update(stage="complete", status="no_new_videos")
                _write_state(paths["state"], run_state)
                return 0

            run_dir.mkdir(parents=True, exist_ok=True)
            config = copy.deepcopy(loaded)
            config["paths"]["input_dir"] = str(run_dir / "input")
            config["paths"]["output_dir"] = str(base_dir / "output")
            config["paths"]["temp_dir"] = str(run_dir / "temp")
            config["paths"]["system_dir"] = str(run_dir / "system")
            config["clipping"]["enable_youtube_downloads"] = True
            config["clipping"]["youtube_links"] = links
            config["clipping"]["channels"] = [channel]
            config["clipping"]["channels_hours_limit"] = hours
            config["clipping"]["user_query"] = str(profile["user_query"])
            config["clipping"]["system_prompt"] = str(profile["system_prompt"])
            config.setdefault("diagnostics", {})["enabled"] = True
            config["diagnostics"]["artifacts_dir"] = str(run_dir / "system" / "diagnostics")

            with _plain_console(run_dir / "engine_console.log") as console:
                engine = AutomationEngine(
                    base_dir=base_dir,
                    config=config,
                    logger=logger,
                    console=console,
                    interactive=False,
                    persist_config=False,
                    state_file=paths["state"],
                    run_state=run_state,
                )
                engine.stop_after_downloads = bool(args.stop_after_downloads)
                engine.required_download_successes = len(links)
                run_state.update(stage="engine_started", engine_step="starting")
                _write_state(paths["state"], run_state)
                engine.run()

            engine_status_path = run_dir / "system" / "status.json"
            engine_status = load_json_file(engine_status_path) if engine_status_path.exists() else {}
            download_summary = run_state.get("download_summary", {})
            failures = int(download_summary.get("failed", 0)) if isinstance(download_summary, dict) else 0
            downloaded = int(download_summary.get("downloaded", 0)) if isinstance(download_summary, dict) else 0
            if args.stop_after_downloads:
                final_status = "downloads_complete"
            elif str(engine_status.get("current_step")) == "completed":
                final_status = "complete"
            elif failures > 0 or str(engine_status.get("current_step")) == "download_failed":
                final_status = "download_failed"
            else:
                final_status = "engine_stopped_early"
            run_state.update(
                stage="complete" if final_status in {"complete", "downloads_complete"} else "failed",
                status=final_status,
                engine_status=engine_status,
            )
            _write_state(paths["state"], run_state)
            return 0 if final_status in {"complete", "downloads_complete"} else 3
    except Exception as exc:
        run_state.update(stage="failed", status="failed", error=str(exc))
        _write_state(paths["state"], run_state)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
