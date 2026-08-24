from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timedelta
import gc
import hashlib
import json
import logging
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Set

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text
import requests

from core.ai_pipeline import AIPipeline, chunk_transcript, estimate_tokens
from core.clipping import (
    assemble_anchor_clips,
    extract_clips,
    scan_input_videos,
    transcribe_video,
)
from core.yt_handler import YTHandler
from ui.components import hard_clear, show_header
from utils.diagnostics import DiagnosticRecorder
from utils.model_selector import ensure_ollama_running, model_billions
from utils.validators import config_paths, load_optional_profile, save_json_file


class _OptionalSpinnerColumn(SpinnerColumn):
    def render(self, task) -> Text:
        if task.fields.get("plain"):
            return Text("")
        return super().render(task)


class _OptionalBarColumn(BarColumn):
    def render(self, task) -> Text:
        if task.fields.get("plain"):
            return Text("")
        return super().render(task)


class _OptionalTaskProgressColumn(TaskProgressColumn):
    def render(self, task) -> Text:
        if task.fields.get("plain"):
            return Text("")
        return super().render(task)


class _OptionalTimeElapsedColumn(TimeElapsedColumn):
    def render(self, task) -> Text:
        if task.fields.get("plain"):
            return Text("")
        return super().render(task)


class _OptionalTimeRemainingColumn(TimeRemainingColumn):
    def render(self, task) -> Text:
        if task.fields.get("plain"):
            return Text("")
        return super().render(task)


class _SettingsChangedError(RuntimeError):
    """Raised when config changes on disk while processing a video."""


class ClippingEngine:
    """Main runtime engine that orchestrates download, transcription, AI scan, and clip extraction."""

    TRANSCRIPT_CHECKPOINT_VERSION = 2

    def __init__(
        self,
        base_dir: Path,
        config: Dict[str, Any],
        logger: logging.Logger,
        console: Console,
        *,
        interactive: bool = True,
        persist_config: bool = True,
    ) -> None:
        self.base_dir = base_dir
        self.config = config
        self.logger = logger
        self.console = console
        self.interactive = bool(interactive)
        self.persist_config = bool(persist_config)
        self.stop_after_downloads = False
        self.required_download_successes = 0
        configured_quality_rank = config.get("clipping", {}).get("max_download_quality_rank")
        maximum_quality_rank = None
        if configured_quality_rank is not None:
            try:
                maximum_quality_rank = int(configured_quality_rank)
            except (TypeError, ValueError):
                maximum_quality_rank = None
        self.yt_handler = YTHandler(
            base_dir=base_dir,
            logger=logger,
            maximum_quality_rank=maximum_quality_rank,
        )

        paths_cfg = config.get("paths", {})
        self.input_dir = base_dir / str(paths_cfg.get("input_dir", "input"))
        self.output_dir = base_dir / str(paths_cfg.get("output_dir", "output"))
        self.temp_dir = base_dir / str(paths_cfg.get("temp_dir", "temp"))
        self.system_dir = base_dir / str(paths_cfg.get("system_dir", "system"))
        self.status_file = self.system_dir / "status.json"
        self.run_cache_dir = self.system_dir / "run_cache"
        self._config_paths = config_paths(self.base_dir)
        self.hardware_profile = load_optional_profile(self._config_paths["hardware_file"])
        self.diagnostics = DiagnosticRecorder.from_config(base_dir=base_dir, config=config, logger=logger)

        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.system_dir.mkdir(parents=True, exist_ok=True)
        self.run_cache_dir.mkdir(parents=True, exist_ok=True)

    def _persist_config(self) -> None:
        if self.persist_config:
            save_json_file(self._config_paths["config_file"], self.config)

    def _write_status(self, payload: Dict[str, Any]) -> None:
        save_json_file(self.status_file, payload)

    def _after_downloads(self, summary: Dict[str, Any]) -> None:
        """Automation hook invoked after the complete YouTube queue is attempted."""

        return None

    def _temp_run_save_enabled(self) -> bool:
        runtime_cfg = self.config.get("runtime", {})
        return bool(runtime_cfg.get("enable_temp_run_save", True))

    @classmethod
    def _current_transcript_meta(cls) -> Dict[str, Any]:
        return {
            "schema_version": int(cls.TRANSCRIPT_CHECKPOINT_VERSION),
            "word_timestamps": True,
        }

    @classmethod
    def _transcript_checkpoint_compatible(cls, payload: Dict[str, Any]) -> bool:
        if "transcript" not in payload:
            return True

        meta = payload.get("transcript_meta")
        if not isinstance(meta, dict):
            return False

        try:
            schema_version = int(meta.get("schema_version", 0))
        except (TypeError, ValueError):
            return False
        if schema_version < cls.TRANSCRIPT_CHECKPOINT_VERSION:
            return False
        return bool(meta.get("word_timestamps"))

    @staticmethod
    def _resume_signature_from_config(config: Dict[str, Any]) -> str:
        clipping = config.get("clipping", {})
        runtime_cfg = config.get("runtime", {})
        ollama_cfg = config.get("ollama", {})
        paths_cfg = config.get("paths", {})

        payload = {
            "pipeline": {
                "clip_assembly_version": 2,
            },
            "clipping": {
                "user_query": str(clipping.get("user_query", "")),
                "system_prompt": str(clipping.get("system_prompt", "")),
                "merge_distance_seconds": int(clipping.get("merge_distance_seconds", 20)),
                "ai_loops": int(clipping.get("ai_loops", 2)),
                "clip_progress_interval": int(clipping.get("clip_progress_interval", 5)),
            },
            "runtime": {
                "total_context_tokens": int(runtime_cfg.get("total_context_tokens", 8192)),
                "max_chunk_tokens": int(runtime_cfg.get("max_chunk_tokens", 6000)),
                "chunk_overlap_segments": int(runtime_cfg.get("chunk_overlap_segments", 3)),
                "enable_bridge_chunks": bool(runtime_cfg.get("enable_bridge_chunks", True)),
                "bridge_chunk_edge_segments": int(runtime_cfg.get("bridge_chunk_edge_segments", 4)),
            },
            "ollama": {
                "model": str(ollama_cfg.get("model", "")),
                "max_output_tokens": int(ollama_cfg.get("max_output_tokens", 900)),
                "context_window": int(ollama_cfg.get("context_window", 8192)),
                "temperature": float(ollama_cfg.get("temperature", 0.2)),
            },
            "paths": {
                "output_dir": str(paths_cfg.get("output_dir", "output")),
            },
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def _current_resume_signature(self) -> str:
        return self._resume_signature_from_config(self.config)

    def _disk_resume_signature(self) -> str:
        if not self.persist_config:
            return self._current_resume_signature()
        on_disk = load_optional_profile(self._config_paths["config_file"])
        if not on_disk:
            return self._current_resume_signature()
        return self._resume_signature_from_config(on_disk)

    def _assert_settings_unchanged(self, expected_signature: str) -> None:
        current_signature = self._disk_resume_signature()
        if current_signature == expected_signature:
            return

        latest = load_optional_profile(self._config_paths["config_file"])
        if latest:
            self.config = latest
        raise _SettingsChangedError(
            "Detected settings change during run. Restarting current video with latest settings."
        )

    @staticmethod
    def _video_fingerprint(video: Path) -> str:
        stat = video.stat()
        payload = f"{video.resolve()}|{int(stat.st_size)}|{int(stat.st_mtime)}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _video_checkpoint_path(self, video: Path, fingerprint: Optional[str] = None) -> Path:
        resolved_fingerprint = str(fingerprint or "").strip()
        if not resolved_fingerprint:
            resolved_fingerprint = self._video_fingerprint(video)
        return self.run_cache_dir / f"{resolved_fingerprint}.json"

    def _load_video_checkpoint(self, video: Path, signature: str) -> Dict[str, Any]:
        if not self._temp_run_save_enabled():
            return {}

        checkpoint_path = self._video_checkpoint_path(video)
        if not checkpoint_path.exists():
            return {}

        payload = load_optional_profile(checkpoint_path)
        if not payload:
            return {}

        expected_fingerprint = self._video_fingerprint(video)
        if str(payload.get("video_fingerprint", "")) != expected_fingerprint:
            return {}
        if str(payload.get("config_signature", "")) != signature:
            return {}
        if bool(payload.get("completed", False)):
            self.logger.info("Discarding completed checkpoint for %s", video.name)
            try:
                checkpoint_path.unlink(missing_ok=True)
            except Exception as exc:
                self.logger.debug("Could not remove completed checkpoint %s: %s", checkpoint_path, exc)
            return {}
        if not self._transcript_checkpoint_compatible(payload):
            self.logger.info(
                "Ignoring outdated transcript checkpoint for %s because it lacks word timestamp metadata.",
                video.name,
            )
            return {}
        return payload

    def _save_video_checkpoint(self, video: Path, payload: Dict[str, Any]) -> None:
        if not self._temp_run_save_enabled():
            return
        fingerprint = str(payload.get("video_fingerprint", "")).strip()
        if not fingerprint:
            try:
                fingerprint = self._video_fingerprint(video)
            except FileNotFoundError:
                self.logger.warning(
                    "Skipping checkpoint save because source file is missing and no prior fingerprint exists: %s",
                    video,
                )
                return
        payload["video_fingerprint"] = fingerprint
        payload["config_signature"] = payload.get("config_signature") or self._current_resume_signature()
        payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
        save_json_file(self._video_checkpoint_path(video, fingerprint=fingerprint), payload)

    def _clear_video_checkpoint(self, video: Path, fingerprint: Optional[str] = None) -> None:
        try:
            self._video_checkpoint_path(video, fingerprint=fingerprint).unlink(missing_ok=True)
        except Exception:
            pass

    def _load_whisper_model(self):
        import whisper
        import torch

        transcription_model = str(self.config["transcription"]["model"]).lower()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info("Loading Whisper model '%s' on %s", transcription_model, device)
        return whisper.load_model(transcription_model, device=device)

    def _ensure_ollama(self) -> bool:
        ollama_url = str(self.config["ollama"]["url"])
        ok = ensure_ollama_running(ollama_url=ollama_url, logger=self.logger)
        if not ok:
            self.logger.error("Ollama did not become ready at %s", ollama_url)
            return False
        return True

    def _download_pending_links(
        self,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Dict[str, Any]:
        enabled = bool(self.config.get("clipping", {}).get("enable_youtube_downloads", True))
        links = list(self.config["clipping"].get("youtube_links", []))
        total_links = len(links)
        summary: Dict[str, Any] = {
            "enabled": enabled,
            "total": total_links,
            "downloaded": 0,
            "already_downloaded": 0,
            "failed": 0,
            "invalid": 0,
            "issues": [],
        }
        if progress_callback:
            progress_callback(0, total_links, "starting")

        if not enabled:
            self.logger.info("YouTube downloading is disabled in settings. Skipping queued links.")
            if progress_callback:
                progress_callback(total_links, total_links, "disabled")
            return summary

        if not links:
            return summary

        diagnostics = self.yt_handler.diagnostics()
        has_cookie_source = bool(diagnostics.get("cookie_file") or diagnostics.get("browser_cookie_source"))
        has_js_runtime = bool(diagnostics.get("js_runtimes"))
        if not has_cookie_source:
            warning = "No cookie source detected. Some YouTube videos may fail."
            if not has_js_runtime:
                warning += " Node/Deno not detected, JS challenge solving is limited."
            summary["issues"].append({"url": "youtube-preflight", "error": warning})

        self.logger.info("Downloading %s queued YouTube links", len(links))
        remaining: List[str] = []
        for index, link in enumerate(links, start=1):
            normalized_link = self.yt_handler.normalize_video_url(str(link))
            if not normalized_link:
                self.logger.warning("Skipping empty/invalid YouTube link: %s", link)
                remaining.append(str(link))
                summary["invalid"] += 1
                summary["failed"] += 1
                summary["issues"].append({"url": str(link), "error": "Invalid YouTube URL or video ID."})
                if progress_callback:
                    progress_callback(index, total_links, "invalid link")
                continue
            try:
                if progress_callback:
                    progress_callback(index - 1, total_links, f"downloading {index}/{total_links}")
                result = self.yt_handler.download_video(url=normalized_link, output_dir=self.input_dir)
            except Exception as exc:
                self.logger.warning("Error downloading %s: %s", normalized_link, exc)
                remaining.append(normalized_link)
                summary["failed"] += 1
                summary["issues"].append({"url": normalized_link, "error": str(exc)})
                if progress_callback:
                    progress_callback(index, total_links, "failed")
                continue

            if result.status == "downloaded":
                summary["downloaded"] += 1
                self.logger.info("Downloaded: %s", result.path)
                if progress_callback:
                    progress_callback(index, total_links, "downloaded")
            elif result.status == "already_downloaded":
                summary["already_downloaded"] += 1
                self.logger.info("Already downloaded, removed from queue: %s", normalized_link)
                if progress_callback:
                    progress_callback(index, total_links, "already downloaded")
            else:
                summary["failed"] += 1
                remaining.append(normalized_link)
                issue = str(result.error or "Unknown download failure.")
                summary["issues"].append({"url": normalized_link, "error": issue})
                self.logger.info("Failed download kept in queue: %s (%s)", normalized_link, issue)
                if progress_callback:
                    progress_callback(index, total_links, "failed")

        self.config["clipping"]["youtube_links"] = remaining
        self._persist_config()
        if progress_callback:
            progress_callback(total_links, total_links, "completed")
        return summary

    def _build_ai_pipeline(self) -> AIPipeline:
        return AIPipeline(
            model=str(self.config["ollama"]["model"]),
            ollama_url=str(self.config["ollama"]["url"]),
            system_prompt=str(self.config["clipping"]["system_prompt"]),
            temperature=float(self.config["ollama"]["temperature"]),
            max_output_tokens=int(self.config["ollama"]["max_output_tokens"]),
            max_context_tokens=int(self.config["runtime"]["total_context_tokens"]),
            logger=self.logger,
            diagnostics=self.diagnostics,
        )

    def _archive_video(self, source_video: Path) -> None:
        target = self.temp_dir / source_video.name
        if target.exists():
            stem = source_video.stem
            suffix = source_video.suffix
            counter = 1
            while True:
                candidate = self.temp_dir / f"{stem}_{counter:03d}{suffix}"
                if not candidate.exists():
                    target = candidate
                    break
                counter += 1
        shutil.move(str(source_video), str(target))

    def _temp_files(self) -> List[Path]:
        return sorted(
            [path for path in self.temp_dir.rglob("*") if path.is_file()],
            key=lambda path: path.stat().st_mtime,
        )

    def _cleanup_temp_by_size(self, max_size_gb: float) -> int:
        max_bytes = int(max(0.1, float(max_size_gb)) * (1024**3))
        files = self._temp_files()
        total_bytes = sum(path.stat().st_size for path in files)
        if total_bytes <= max_bytes:
            return 0

        deleted = 0
        for file_path in files:
            if total_bytes <= max_bytes:
                break
            try:
                file_size = file_path.stat().st_size
                file_path.unlink(missing_ok=True)
                total_bytes -= file_size
                deleted += 1
            except Exception as exc:
                self.logger.debug("Temp cleanup could not remove %s: %s", file_path, exc)
        return deleted

    def _cleanup_temp_by_age(self, max_age_days: int) -> int:
        cutoff = datetime.now() - timedelta(days=max(1, int(max_age_days)))
        deleted = 0
        for file_path in self._temp_files():
            try:
                modified = datetime.fromtimestamp(file_path.stat().st_mtime)
            except Exception:
                continue
            if modified >= cutoff:
                continue
            try:
                file_path.unlink(missing_ok=True)
                deleted += 1
            except Exception as exc:
                self.logger.debug("Temp cleanup could not remove %s: %s", file_path, exc)
        return deleted

    def _apply_temp_cleanup_policy(self) -> None:
        cleanup = self.config.get("maintenance", {}).get("temp_cleanup", {})
        if not isinstance(cleanup, dict):
            return

        mode = str(cleanup.get("mode", "never")).strip().lower()
        if mode == "never":
            return

        deleted = 0
        if mode == "max_size_gb":
            deleted = self._cleanup_temp_by_size(float(cleanup.get("max_size_gb", 20)))
        elif mode == "max_age_days":
            deleted = self._cleanup_temp_by_age(int(cleanup.get("max_age_days", 30)))

        if deleted > 0:
            self.logger.info("Temp cleanup removed %s file(s) using mode '%s'.", deleted, mode)

    @contextmanager
    def _mute_console_logs(self) -> Iterator[None]:
        root_logger = logging.getLogger()
        handler_levels: List[tuple[logging.Handler, int]] = []
        for handler in root_logger.handlers:
            stream = getattr(handler, "stream", None)
            if stream in (sys.stdout, sys.stderr):
                handler_levels.append((handler, handler.level))
                handler.setLevel(logging.CRITICAL + 1)
        try:
            yield
        finally:
            for handler, level in handler_levels:
                handler.setLevel(level)

    @staticmethod
    def _extract_min_duration_seconds(user_query: str) -> float:
        text = str(user_query or "").lower()
        patterns = (
            (r"(?:at\s*least|atleast|minimum|min(?:imum)?|no\s+less\s+than)\s*(\d+(?:\.\d+)?)\s*(seconds?|secs?|sec|s)\b", 1.0),
            (r"(?:at\s*least|atleast|minimum|min(?:imum)?|no\s+less\s+than)\s*(\d+(?:\.\d+)?)\s*(minutes?|mins?|min|m)\b", 60.0),
            (r"(\d+(?:\.\d+)?)\s*(seconds?|secs?|sec|s)\s*(?:minimum|min|or\s+more|\+)", 1.0),
            (r"(\d+(?:\.\d+)?)\s*(minutes?|mins?|min|m)\s*(?:minimum|min|or\s+more|\+)", 60.0),
        )
        for pattern, scale in patterns:
            match = re.search(pattern, text)
            if not match:
                continue
            try:
                value = float(match.group(1))
            except Exception:
                continue
            if value > 0:
                return value * scale
        return 0.0

    @classmethod
    def _build_effective_query(cls, user_query: str) -> tuple[str, float]:
        min_duration = cls._extract_min_duration_seconds(user_query)

        constraints: List[str] = []
        if min_duration > 0:
            constraints.append(f"- Minimum clip duration: {min_duration:.1f} seconds.")
            constraints.append("- 'At least' is a lower bound, not an exact target duration.")
            constraints.append("- Prefer longer continuous relevant sections when they remain on-topic.")
            constraints.append("- Avoid cutting mid-sentence; keep natural spoken boundaries.")

        if not constraints:
            return str(user_query), min_duration

        constrained_query = (
            f"{str(user_query).strip()}\n\n"
            "Strict constraints:\n"
            f"{chr(10).join(constraints)}"
        )
        return constrained_query, min_duration

    @staticmethod
    def _filter_min_duration(
        clips: List[List[float]],
        min_duration_seconds: float,
    ) -> tuple[List[List[float]], int]:
        if min_duration_seconds <= 0:
            return clips, 0
        filtered: List[List[float]] = []
        rejected = 0
        for clip in clips:
            if len(clip) < 3:
                continue
            start = float(clip[0])
            end = float(clip[1])
            score = float(clip[2])
            if (end - start) < min_duration_seconds:
                rejected += 1
                continue
            filtered.append([start, end, score])
        return filtered, rejected

    @staticmethod
    def _estimate_transcription_progress(elapsed_seconds: float) -> tuple[int, str]:
        # Estimated progress to keep UI alive while whisper transcribes in one blocking call.
        if elapsed_seconds < 1.6:
            stage = "loading audio"
        elif elapsed_seconds < 5.0:
            stage = "running whisper"
        elif elapsed_seconds < 12.0:
            stage = "decoding speech"
        else:
            stage = "finalizing transcript"

        progress_percent = int(min(96.0, 8.0 + (elapsed_seconds * 3.1)))
        return max(1, progress_percent), stage

    def _render_engine_frame(self, subtitle: Optional[str] = None) -> None:
        hard_clear(self.console)
        show_header(self.console)
        self.console.rule("[bold cyan]Clipping Engine")
        if subtitle:
            self.console.print(f"[cyan]{subtitle}[/cyan]")

    def _show_youtube_download_issues(self, download_summary: Dict[str, Any]) -> None:
        issues = list(download_summary.get("issues", []))
        if not issues:
            return

        lines = [
            "[bold yellow]YouTube download issues were detected.[/bold yellow]",
            f"- Failed links kept in queue: {int(download_summary.get('failed', 0))}",
            f"- Invalid links kept in queue: {int(download_summary.get('invalid', 0))}",
            "",
            "Top failures:",
        ]
        for item in issues[:5]:
            url = str(item.get("url", "")).strip()
            error = str(item.get("error", "")).strip()
            lines.append(f"- {url or '<unknown>'}")
            lines.append(f"  {error or 'Unknown error'}")
        if len(issues) > 5:
            lines.append(f"- ... and {len(issues) - 5} more")

        self.console.print(
            Panel(
                "\n".join(lines),
                title="YouTube Warning",
                border_style="yellow",
            )
        )
        if not self.interactive:
            self.logger.warning(
                "YouTube download issues recorded during a non-interactive run; continuing without prompting."
            )
            return
        input("Press Enter to continue pipeline run...")

    def _resolve_runtime_chunk_cap(
        self,
        model_name: str,
        configured_chunk_tokens: int,
    ) -> tuple[int, float]:
        """
        Preserve the configured transcript context while reporting hardware pressure.

        Context length is bounded separately by `_resolve_prompt_aware_chunk_cap`, which
        reserves prompt and output space inside the model's real context window. Reducing
        transcript context based on model size or VRAM pressure does not reduce the loaded
        model's memory footprint and materially harms long-form clip selection.

        Returns: (effective_chunk_tokens, pressure_ratio)
        pressure_ratio > 1 means model estimate exceeds budget.
        """
        billions = model_billions(model_name)
        gpu_vram_gb = float(self.hardware_profile.get("gpu_vram_gb", 0.0) or 0.0)
        ram_gb = float(self.hardware_profile.get("ram_gb", 0.0) or 0.0)
        intensity = str(self.config.get("runtime", {}).get("setup_intensity", "balanced")).lower().strip()
        if gpu_vram_gb > 0:
            multiplier = {"light": 0.45, "balanced": 0.64, "maximum": 0.82}.get(intensity, 0.64)
            budget_gb = max(2.0, gpu_vram_gb * multiplier)
        else:
            multiplier = {"light": 0.18, "balanced": 0.28, "maximum": 0.40}.get(intensity, 0.28)
            budget_gb = max(2.0, ram_gb * multiplier)

        estimated_model_gb = max(1.5, (billions or 10.0) * 0.62)
        pressure_ratio = estimated_model_gb / max(1.0, budget_gb)
        return max(900, int(configured_chunk_tokens)), float(pressure_ratio)

    @staticmethod
    def _resolve_prompt_aware_chunk_cap(
        configured_chunk_tokens: int,
        total_context_tokens: int,
        max_output_tokens: int,
        user_query: str,
        system_prompt: str,
    ) -> tuple[int, int, int]:
        # Reserve room for prompts, policy guardrails, and response framing.
        prompt_tokens = (
            estimate_tokens(str(user_query or ""))
            + estimate_tokens(str(system_prompt or ""))
            + 700
        )
        available = int(total_context_tokens) - int(max_output_tokens) - int(prompt_tokens)
        if available >= 900:
            effective = min(int(configured_chunk_tokens), int(available))
            return max(900, int(effective)), prompt_tokens, int(available)

        # Keep engine functional for very large prompts by allowing smaller chunk caps.
        effective = max(256, min(int(configured_chunk_tokens), int(max(256, available))))
        return int(effective), prompt_tokens, int(available)

    def _runtime_video_parameters(self) -> Dict[str, Any]:
        clipping_cfg = self.config.get("clipping", {})
        runtime_cfg = self.config.get("runtime", {})
        ollama_cfg = self.config.get("ollama", {})

        merge_distance = float(clipping_cfg.get("merge_distance_seconds", 20))
        ai_loops = int(clipping_cfg.get("ai_loops", 2))
        user_query = str(clipping_cfg.get("user_query", ""))
        system_prompt = str(clipping_cfg.get("system_prompt", ""))
        effective_query, min_duration_seconds = self._build_effective_query(user_query)

        model_name = str(ollama_cfg.get("model", ""))
        configured_chunk_tokens = int(runtime_cfg.get("max_chunk_tokens", 6000))
        total_context_tokens = int(runtime_cfg.get("total_context_tokens", 8192))
        max_output_tokens = int(ollama_cfg.get("max_output_tokens", 900))
        chunk_overlap_segments = int(runtime_cfg.get("chunk_overlap_segments", 3))
        enable_bridge_chunks = bool(runtime_cfg.get("enable_bridge_chunks", True))
        bridge_chunk_edge_segments = int(runtime_cfg.get("bridge_chunk_edge_segments", 4))
        progress_interval = int(clipping_cfg.get("clip_progress_interval", 5))

        runtime_capped_chunk_tokens, pressure_ratio = self._resolve_runtime_chunk_cap(
            model_name=model_name,
            configured_chunk_tokens=configured_chunk_tokens,
        )
        prompt_capped_chunk_tokens, prompt_tokens, prompt_available = self._resolve_prompt_aware_chunk_cap(
            configured_chunk_tokens=runtime_capped_chunk_tokens,
            total_context_tokens=total_context_tokens,
            max_output_tokens=max_output_tokens,
            user_query=user_query,
            system_prompt=system_prompt,
        )
        max_chunk_tokens = min(runtime_capped_chunk_tokens, prompt_capped_chunk_tokens)

        return {
            "merge_distance": merge_distance,
            "ai_loops": ai_loops,
            "user_query": user_query,
            "system_prompt": system_prompt,
            "effective_query": effective_query,
            "min_duration_seconds": min_duration_seconds,
            "model_name": model_name,
            "configured_chunk_tokens": configured_chunk_tokens,
            "chunk_overlap_segments": chunk_overlap_segments,
            "enable_bridge_chunks": enable_bridge_chunks,
            "bridge_chunk_edge_segments": bridge_chunk_edge_segments,
            "progress_interval": progress_interval,
            "runtime_capped_chunk_tokens": runtime_capped_chunk_tokens,
            "pressure_ratio": pressure_ratio,
            "prompt_tokens": prompt_tokens,
            "prompt_available": prompt_available,
            "max_chunk_tokens": max_chunk_tokens,
        }

    @staticmethod
    def _run_with_activity_heartbeat(
        progress: Progress,
        activity_task_id: Any,
        base_description: str,
        action: Callable[[], Any],
        interval_seconds: float = 0.25,
        heartbeat_callback: Optional[Callable[[float, int], None]] = None,
    ) -> Any:
        started_at = time.monotonic()
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(action)
            pulse = 0
            while not future.done():
                time.sleep(max(0.05, float(interval_seconds)))
                pulse = (pulse + 1) % 4
                elapsed = max(0.0, time.monotonic() - started_at)
                if heartbeat_callback is not None:
                    heartbeat_callback(elapsed, pulse)
                else:
                    dots = "." * pulse
                    progress.update(activity_task_id, description=f"{base_description}{dots}")
                    progress.refresh()
            return future.result()

    @staticmethod
    def _augment_chunks_with_bridges(
        chunks: List[List[List[Any]]],
        max_bridge_segments: int = 4,
    ) -> tuple[List[List[List[Any]]], int]:
        if len(chunks) <= 1:
            return chunks, 0

        edge_segments = max(1, int(max_bridge_segments))
        bridges: List[List[List[Any]]] = []
        for index in range(len(chunks) - 1):
            left = chunks[index][-edge_segments:]
            right = chunks[index + 1][:edge_segments]
            bridge = [list(item) for item in (left + right) if len(item) >= 3]
            if len(bridge) >= 2:
                bridges.append(bridge)

        combined = chunks + bridges
        deduped: List[List[List[Any]]] = []
        seen: set[tuple[float, float, int]] = set()
        for chunk in combined:
            if not chunk:
                continue
            try:
                key = (round(float(chunk[0][0]), 3), round(float(chunk[-1][1]), 3), len(chunk))
            except Exception:
                key = (0.0, 0.0, len(chunk))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(chunk)
        return deduped, len(bridges)

    def _offload_ollama_model(self) -> None:
        model_name = str(self.config.get("ollama", {}).get("model", "")).strip()
        ollama_url = str(self.config.get("ollama", {}).get("url", "http://localhost:11434")).rstrip("/")
        if not model_name:
            return

        try:
            ps_response = requests.get(f"{ollama_url}/api/ps", timeout=10)
            loaded_models = []
            if ps_response.ok:
                payload = ps_response.json()
                loaded_models = [str(item.get("model", "")).lower() for item in payload.get("models", [])]
            if model_name.lower() not in loaded_models:
                self.logger.info("Ollama model `%s` is not currently loaded; no offload needed.", model_name)
                return
        except Exception as exc:
            self.logger.debug("Could not query loaded Ollama models: %s", exc)

        try:
            subprocess.run(
                ["ollama", "stop", model_name],
                check=False,
                capture_output=True,
                timeout=20,
            )
            self.logger.info("Offloaded Ollama model `%s` via `ollama stop`.", model_name)
            return
        except Exception as exc:
            self.logger.debug("Fallback `ollama stop` failed: %s", exc)

        try:
            payload = {
                "model": model_name,
                "prompt": "",
                "stream": False,
                "keep_alive": 0,
            }
            response = requests.post(
                f"{ollama_url}/api/generate",
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=20,
            )
            if response.ok:
                self.logger.info("Offloaded Ollama model `%s` via keep_alive=0.", model_name)
        except Exception as exc:
            self.logger.debug("Ollama API keep_alive unload failed: %s", exc)

    def _offload_runtime_resources(self, whisper_model: Any) -> None:
        try:
            if whisper_model is not None:
                del whisper_model
            gc.collect()
        except Exception:
            pass

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
                self.logger.info("Cleared CUDA cache after engine run.")
        except Exception:
            pass

        self._offload_ollama_model()

    def run(self) -> None:
        self._render_engine_frame()
        diagnostics_cfg = self.config.get("diagnostics", {})
        requested_run_id = ""
        if isinstance(diagnostics_cfg, dict):
            requested_run_id = str(diagnostics_cfg.get("run_id", "")).strip()
        run_id = requested_run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
        self.logger.info("Clipping engine run initialized (run_id=%s)", run_id)
        if self.diagnostics.enabled:
            self.diagnostics.record_run_metadata(
                {
                    "run_id": run_id,
                    "model": str(self.config.get("ollama", {}).get("model", "")),
                    "user_query": str(self.config.get("clipping", {}).get("user_query", "")),
                    "system_prompt": str(self.config.get("clipping", {}).get("system_prompt", "")),
                    "original_video_path": str(diagnostics_cfg.get("original_video_path", "")) if isinstance(diagnostics_cfg, dict) else "",
                    "scratch_input_copy_path": str(diagnostics_cfg.get("scratch_input_copy_path", "")) if isinstance(diagnostics_cfg, dict) else "",
                    "artifacts_dir": str(getattr(self.diagnostics, "artifacts_dir", "") or ""),
                }
            )

        status: Dict[str, Any] = {
            "run_id": run_id,
            "total_videos": 0,
            "progress_percent": 0,
            "current_video": None,
            "current_step": "starting",
        }
        run_stats: Dict[str, int] = {
            "ai_clipping_base_chunks": 0,
            "ai_clipping_bridge_chunks": 0,
            "ai_clipping_chunks_scanned": 0,
            "ai_clipping_candidates": 0,
            "merged_candidates": 0,
            "clips_extracted": 0,
        }

        whisper_model: Any = None
        try:
            with self._mute_console_logs():
                with Progress(
                    _OptionalSpinnerColumn(style="cyan"),
                    TextColumn("[progress.description]{task.description}"),
                    _OptionalBarColumn(bar_width=42),
                    _OptionalTaskProgressColumn(),
                    _OptionalTimeElapsedColumn(),
                    _OptionalTimeRemainingColumn(),
                    console=self.console,
                    transient=True,
                    ) as progress:
                    queued_links = list(self.config["clipping"].get("youtube_links", []))
                    queued_count = len(queued_links)
                    links_task = progress.add_task(
                        f"[yellow]YouTube links: 0/{queued_count}",
                        total=max(1, queued_count),
                        completed=0,
                    )
                    videos_task = progress.add_task(
                        "[cyan]Videos: waiting",
                        total=1,
                        completed=0,
                    )
                    activity_task = progress.add_task(
                        "[green]Activity: Booting",
                        total=1,
                        completed=0,
                    )
                    chunk_progress_task = progress.add_task(
                        "[magenta]Chunks: waiting",
                        total=1,
                        completed=0,
                        start=False,
                        visible=False,
                    )
                    ai_loop_task = progress.add_task(
                        "[magenta]AI loops: waiting",
                        total=1,
                        completed=0,
                        start=False,
                        visible=False,
                    )
                    activity_panel_title_task = progress.add_task(
                        "[bright_green]────────────[/bright_green][bold bright_cyan] Activity Panel [/bold bright_cyan][bright_green]────────────[/bright_green]",
                        total=1,
                        completed=1,
                        plain=True,
                    )
                    panel_tasks = {
                        "ai_clipping_base_chunks": progress.add_task(
                            "[bright_green]Base chunks: 0[/bright_green]",
                            total=1,
                            completed=1,
                            plain=True,
                        ),
                        "ai_clipping_bridge_chunks": progress.add_task(
                            "[bright_green]Bridge chunks: 0[/bright_green]",
                            total=1,
                            completed=1,
                            plain=True,
                        ),
                        "ai_clipping_chunks_scanned": progress.add_task(
                            "[bright_green]Chunks scanned: 0[/bright_green]",
                            total=1,
                            completed=1,
                            plain=True,
                        ),
                        "ai_clipping_candidates": progress.add_task(
                            "[bright_green]Clips found: 0[/bright_green]",
                            total=1,
                            completed=1,
                            plain=True,
                        ),
                        "merged_candidates": progress.add_task(
                            "[bright_green]Candidates after merge: 0[/bright_green]",
                            total=1,
                            completed=1,
                            plain=True,
                        ),
                        "clips_extracted": progress.add_task(
                            "[bright_green]Clips extracted: 0[/bright_green]",
                            total=1,
                            completed=1,
                            plain=True,
                        ),
                        "output_folder": progress.add_task(
                            f"[bright_green]Output folder: {self.output_dir}[/bright_green]",
                            total=1,
                            completed=1,
                            plain=True,
                        ),
                    }

                    def refresh_activity_panel() -> None:
                        bridge_chunks_enabled = bool(self.config.get("runtime", {}).get("enable_bridge_chunks", True))
                        progress.update(
                            activity_panel_title_task,
                            description=(
                                "[bright_green]━━━━━━━━━━━━[/bright_green]"
                                "[bold bright_cyan] Activity Panel [/bold bright_cyan]"
                                "[bright_green]━━━━━━━━━━━━[/bright_green]"
                            ),
                        )
                        progress.update(
                            panel_tasks["ai_clipping_base_chunks"],
                            description=f"[bright_green]Base chunks: {run_stats['ai_clipping_base_chunks']}[/bright_green]",
                        )
                        progress.update(
                            panel_tasks["ai_clipping_bridge_chunks"],
                            description=f"[bright_green]Bridge chunks: {run_stats['ai_clipping_bridge_chunks']}[/bright_green]",
                            visible=bridge_chunks_enabled,
                        )
                        progress.update(
                            panel_tasks["ai_clipping_chunks_scanned"],
                            description=f"[bright_green]Chunks scanned: {run_stats['ai_clipping_chunks_scanned']}[/bright_green]",
                        )
                        progress.update(
                            panel_tasks["ai_clipping_candidates"],
                            description=f"[bright_green]Clips found: {run_stats['ai_clipping_candidates']}[/bright_green]",
                        )
                        progress.update(
                            panel_tasks["merged_candidates"],
                            description=f"[bright_green]Candidates after merge: {run_stats['merged_candidates']}[/bright_green]",
                        )
                        progress.update(
                            panel_tasks["clips_extracted"],
                            description=f"[bright_green]Clips extracted: {run_stats['clips_extracted']}[/bright_green]",
                        )
                        progress.update(
                            panel_tasks["output_folder"],
                            description=f"[bright_green]Output folder: {self.output_dir}[/bright_green]",
                        )
                        progress.refresh()

                    refresh_activity_panel()
                    self._write_status(status)
                    status["current_step"] = "checking_ollama"
                    self._write_status(status)
                    progress.update(activity_task, description="[green]Activity: Checking Ollama", total=1, completed=0)
                    progress.refresh()
                    if not self._ensure_ollama():
                        self.console.print("[bold red]Ollama is not available. Aborting engine run.[/bold red]")
                        return
                    progress.update(activity_task, completed=1)
    
                    def on_download_progress(completed: int, total: int, state: str) -> None:
                        display_total = max(1, total)
                        display_completed = min(max(0, completed), display_total)
                        if total <= 0:
                            links_label = "[yellow]YouTube links: 0/0 (none queued)"
                            activity_label = "[green]Activity: Downloading YouTube links (none queued)"
                        else:
                            links_label = f"[yellow]YouTube links: {min(completed, total)}/{total} ({state})"
                            activity_label = (
                                f"[green]Activity: Downloading YouTube links {min(completed, total)}/{total}"
                            )
                        progress.update(
                            links_task,
                            total=display_total,
                            completed=display_completed,
                            description=links_label,
                        )
                        progress.update(
                            activity_task,
                            total=display_total,
                            completed=display_completed,
                            description=activity_label,
                        )
                        progress.refresh()
    
                    status["current_step"] = "downloading_links"
                    self._write_status(status)
                    download_summary = self._download_pending_links(progress_callback=on_download_progress)
                    self._after_downloads(download_summary)
                    successful_downloads = int(download_summary.get("downloaded", 0)) + int(
                        download_summary.get("already_downloaded", 0)
                    )
                    if (
                        self.required_download_successes > 0
                        and successful_downloads < self.required_download_successes
                    ):
                        status["current_step"] = "download_failed"
                        self._write_status(status)
                        return
                    if self.stop_after_downloads:
                        status["current_step"] = "downloads_complete"
                        self._write_status(status)
                        return
                    if queued_count == 0:
                        progress.update(links_task, total=1, completed=1, description="[yellow]YouTube links: 0/0")
                        progress.update(
                            activity_task,
                            total=1,
                            completed=1,
                            description="[green]Activity: No YouTube downloads queued",
                        )
                    elif not bool(download_summary.get("enabled", True)):
                        progress.update(
                            links_task,
                            total=max(1, queued_count),
                            completed=max(1, queued_count),
                            description=f"[yellow]YouTube links: {queued_count}/{queued_count} (disabled)",
                        )
                        progress.update(
                            activity_task,
                            total=1,
                            completed=1,
                            description="[green]Activity: YouTube downloading disabled",
                        )

                    self._apply_temp_cleanup_policy()
                    self._show_youtube_download_issues(download_summary)
    
                    status["current_step"] = "scanning_input"
                    self._write_status(status)
                    progress.update(activity_task, description="[green]Activity: Scanning input videos", total=1, completed=0)
                    progress.refresh()
                    videos = scan_input_videos(self.input_dir)
                    progress.update(activity_task, completed=1)
                    if not videos:
                        progress.update(videos_task, description="[cyan]Videos: 0/0", total=1, completed=1)
                        self.console.print("[yellow]No videos found in input/. Add files or fetch links first.[/yellow]")
                        return
    
                    status["total_videos"] = len(videos)
                    status["progress_percent"] = 0
                    self._write_status(status)
                    progress.update(videos_task, description=f"[cyan]Videos: 0/{len(videos)}", total=len(videos), completed=0)
    
                    status["current_step"] = "loading_whisper"
                    self._write_status(status)
                    progress.update(activity_task, description="[green]Activity: Loading Whisper model", total=1, completed=0)
                    progress.refresh()
                    try:
                        whisper_model = self._load_whisper_model()
                    except Exception as exc:
                        self.logger.error("Could not load Whisper model: %s", exc)
                        self.console.print(f"[red]Whisper load failed: {exc}[/red]")
                        return
                    progress.update(activity_task, completed=1)
    
                    ai_pipeline = self._build_ai_pipeline()
                    initial_runtime = self._runtime_video_parameters()
                    if initial_runtime["min_duration_seconds"] > 0:
                        self.logger.info(
                            "Applying minimum clip duration: %.1f seconds",
                            initial_runtime["min_duration_seconds"],
                        )
                    if initial_runtime["runtime_capped_chunk_tokens"] != initial_runtime["configured_chunk_tokens"]:
                        self.logger.info(
                            "Adjusted chunk token cap from %s to %s for model/hardware pressure (ratio=%.2f).",
                            initial_runtime["configured_chunk_tokens"],
                            initial_runtime["runtime_capped_chunk_tokens"],
                            initial_runtime["pressure_ratio"],
                        )
                    if initial_runtime["max_chunk_tokens"] != initial_runtime["runtime_capped_chunk_tokens"]:
                        self.logger.info(
                            (
                                "Adjusted chunk token cap from %s to %s for prompt pressure "
                                "(prompt_tokens=%s available_context=%s)."
                            ),
                            initial_runtime["runtime_capped_chunk_tokens"],
                            initial_runtime["max_chunk_tokens"],
                            initial_runtime["prompt_tokens"],
                            initial_runtime["prompt_available"],
                        )
                    config_signature = self._current_resume_signature()
    
                    for index, video in enumerate(videos, start=1):
                        progress.update(
                            videos_task,
                            description=f"[cyan]Videos: {index - 1}/{len(videos)}",
                        )
                        progress.reset(activity_task, total=7, completed=0, start=True)
                        progress.update(
                            chunk_progress_task,
                            visible=False,
                            total=1,
                            completed=0,
                            description="[magenta]Chunks: waiting",
                        )
                        progress.update(
                            ai_loop_task,
                            visible=False,
                            total=1,
                            completed=0,
                            description="[magenta]AI loops: waiting",
                        )
                        progress.update(
                            activity_task,
                            description=f"[green]Activity: Preparing {index}/{len(videos)} {video.name}",
                        )

                        status["current_video"] = video.name
                        status["progress_percent"] = round(((index - 1) / len(videos)) * 100, 1)
                        if self.diagnostics.enabled:
                            self.diagnostics.record_video_metadata(
                                video.name,
                                {
                                    "video_index": index,
                                    "total_videos": len(videos),
                                    "source_video_path": str(video),
                                },
                            )
                        processed_video = False
                        while not processed_video:
                            config_signature = self._current_resume_signature()
                            runtime = self._runtime_video_parameters()
                            merge_distance = float(runtime["merge_distance"])
                            ai_loops = int(runtime["ai_loops"])
                            effective_query = str(runtime["effective_query"])
                            min_duration_seconds = float(runtime["min_duration_seconds"])
                            max_chunk_tokens = int(runtime["max_chunk_tokens"])
                            chunk_overlap_segments = int(runtime["chunk_overlap_segments"])
                            enable_bridge_chunks = bool(runtime["enable_bridge_chunks"])
                            bridge_chunk_edge_segments = int(runtime["bridge_chunk_edge_segments"])
                            progress_interval = int(runtime["progress_interval"])
                            checkpoint = self._load_video_checkpoint(video=video, signature=config_signature)
                            checkpoint["config_signature"] = config_signature
                            status["current_step"] = "transcribing"
                            self._write_status(status)
                            progress.update(activity_task, description="[green]Activity: Transcribing")
                            progress.refresh()

                            try:
                                self._assert_settings_unchanged(config_signature)
                                transcript = checkpoint.get("transcript")
                                used_transcript_checkpoint = isinstance(transcript, list)
                                if not isinstance(transcript, list):
                                    transcript = transcribe_video(
                                        video_path=video,
                                        whisper_model=whisper_model,
                                        logger=self.logger,
                                    )
                                    checkpoint["transcript"] = transcript
                                    checkpoint["transcript_meta"] = self._current_transcript_meta()
                                    self._save_video_checkpoint(video, checkpoint)
                                else:
                                    self.logger.info("Using saved transcript checkpoint for %s", video.name)
                                if self.diagnostics.enabled:
                                    self.diagnostics.record_transcript(
                                        video.name,
                                        transcript,
                                        used_checkpoint=used_transcript_checkpoint,
                                    )
                                progress.advance(activity_task)
                                if not transcript:
                                    self.logger.warning("Empty transcript for %s", video.name)
                                    processed_video = True
                                    break

                                self._assert_settings_unchanged(config_signature)
                                status["current_step"] = "chunking"
                                self._write_status(status)
                                progress.update(activity_task, description="[green]Activity: Chunking transcript")
                                progress.refresh()
                                chunking_state = checkpoint.get("chunking")
                                chunking_meta = checkpoint.get("chunking_meta", {})
                                expected_meta = {
                                    "max_chunk_tokens": int(max_chunk_tokens),
                                    "chunk_overlap_segments": int(max(0, chunk_overlap_segments)),
                                    "enable_bridge_chunks": bool(enable_bridge_chunks),
                                    "bridge_chunk_edge_segments": int(max(1, bridge_chunk_edge_segments)),
                                }
                                used_chunk_checkpoint = False
                                if (
                                    isinstance(chunking_state, dict)
                                    and chunking_meta == expected_meta
                                    and isinstance(chunking_state.get("base_chunks"), list)
                                    and isinstance(chunking_state.get("chunks"), list)
                                ):
                                    base_chunks = chunking_state.get("base_chunks", [])
                                    chunks = chunking_state.get("chunks", [])
                                    bridge_count = int(chunking_state.get("bridge_count", 0))
                                    used_chunk_checkpoint = True
                                    self.logger.info("Using saved chunk checkpoint for %s", video.name)
                                else:
                                    base_chunks = chunk_transcript(
                                        transcript=transcript,
                                        max_tokens=max_chunk_tokens,
                                        overlap_segments=max(0, chunk_overlap_segments),
                                    )
                                    if enable_bridge_chunks:
                                        chunks, bridge_count = self._augment_chunks_with_bridges(
                                            base_chunks,
                                            max_bridge_segments=max(1, bridge_chunk_edge_segments),
                                        )
                                    else:
                                        chunks, bridge_count = base_chunks, 0
                                    checkpoint["chunking"] = {
                                        "base_chunks": base_chunks,
                                        "chunks": chunks,
                                        "bridge_count": int(bridge_count),
                                    }
                                    checkpoint["chunking_meta"] = expected_meta
                                    self._save_video_checkpoint(video, checkpoint)
                                if self.diagnostics.enabled:
                                    self.diagnostics.record_chunking(
                                        video.name,
                                        base_chunks=base_chunks,
                                        scanned_chunks=chunks,
                                        bridge_count=bridge_count,
                                        meta=expected_meta,
                                        used_checkpoint=used_chunk_checkpoint,
                                    )
                                run_stats["ai_clipping_base_chunks"] += len(base_chunks)
                                run_stats["ai_clipping_bridge_chunks"] += int(bridge_count)
                                refresh_activity_panel()
                                progress.advance(activity_task)
                                if not chunks:
                                    self.logger.warning("No chunks created for %s", video.name)
                                    processed_video = True
                                    break

                                self._assert_settings_unchanged(config_signature)
                                status["current_step"] = "ai_scanning"
                                self._write_status(status)
                                total_chunks = len(chunks)
                                loop_count = max(1, int(ai_loops))
                                progress.reset(chunk_progress_task, total=total_chunks, completed=0, start=True)
                                progress.reset(ai_loop_task, total=loop_count, completed=0, start=True)
                                progress.update(chunk_progress_task, visible=True, description=f"[magenta]Chunks: 0/{total_chunks}")
                                progress.update(
                                    ai_loop_task,
                                    visible=True,
                                    description=f"[magenta]AI loops: 0/{loop_count} (chunk 1/{total_chunks})",
                                )
                                progress.update(activity_task, description=f"[green]Activity: Scanning chunks 0/{total_chunks}")
                                progress.refresh()

                                ai_scan_state = checkpoint.get("ai_scan")
                                if not isinstance(ai_scan_state, dict):
                                    ai_scan_state = {}
                                chunk_outputs = ai_scan_state.get("chunk_outputs")
                                if not isinstance(chunk_outputs, dict):
                                    chunk_outputs = {}

                                raw_clip_groups: List[List[List[float]]] = []
                                raw_candidate_details: List[Dict[str, Any]] = []
                                candidate_counter = 0
                                for chunk_index, chunk in enumerate(chunks, start=1):
                                    progress.update(
                                        chunk_progress_task,
                                        completed=chunk_index - 1,
                                        description=f"[magenta]Chunks: {chunk_index - 1}/{total_chunks}",
                                    )
                                    progress.update(
                                        activity_task,
                                        description=f"[green]Activity: Scanning chunks {chunk_index}/{total_chunks}",
                                    )
                                    progress.update(
                                        ai_loop_task,
                                        completed=0,
                                        description=f"[magenta]AI loops: 0/{loop_count} (chunk {chunk_index}/{total_chunks})",
                                    )
                                    loop_outputs = chunk_outputs.get(str(chunk_index))
                                    if not isinstance(loop_outputs, dict):
                                        loop_outputs = {}
                                    combined: List[List[float]] = []

                                    for loop_index in range(1, loop_count + 1):
                                        self._assert_settings_unchanged(config_signature)
                                        progress.update(
                                            ai_loop_task,
                                            completed=loop_index - 1,
                                            description=(
                                                f"[magenta]AI loops: {loop_index - 1}/{loop_count} "
                                                f"(chunk {chunk_index}/{total_chunks})"
                                            ),
                                        )
                                        cached = loop_outputs.get(str(loop_index))
                                        if isinstance(cached, list):
                                            output = cached
                                            if self.diagnostics.enabled:
                                                self.diagnostics.record_scan_attempt(
                                                    video.name,
                                                    {
                                                        "chunk_index": chunk_index,
                                                        "loop_index": loop_index,
                                                        "attempt_index": 0,
                                                        "strict_mode": False,
                                                        "status": "cached_output",
                                                        "error": "",
                                                        "raw_response": "",
                                                        "parsed_output": [list(row) for row in cached],
                                                        "chunk": [list(row) for row in chunk],
                                                        "prompt": "",
                                                        "system_message": "",
                                                        "request_payload": {},
                                                    },
                                                )
                                        else:
                                            output = ai_pipeline.scan_chunk_with_retries(
                                                chunk=chunk,
                                                user_query=effective_query,
                                                diagnostic_context={
                                                    "video_name": video.name,
                                                    "chunk_index": chunk_index,
                                                    "loop_index": loop_index,
                                                },
                                            )
                                            loop_outputs[str(loop_index)] = output
                                            chunk_outputs[str(chunk_index)] = loop_outputs
                                            ai_scan_state["chunk_outputs"] = chunk_outputs
                                            checkpoint["ai_scan"] = ai_scan_state
                                            self._save_video_checkpoint(video, checkpoint)
                                        chunk_start = float(chunk[0][0]) if chunk else 0.0
                                        chunk_end = float(chunk[-1][1]) if chunk else 0.0
                                        chunk_first_end = float(chunk[0][1]) if chunk else chunk_start
                                        chunk_last_start = float(chunk[-1][0]) if chunk else chunk_end
                                        for clip_row in output:
                                            if len(clip_row) < 3:
                                                continue
                                            candidate_counter += 1
                                            raw_candidate_details.append(
                                                {
                                                    "candidate_id": candidate_counter,
                                                    "chunk_index": chunk_index,
                                                    "loop_index": loop_index,
                                                    "start": float(clip_row[0]),
                                                    "end": float(clip_row[1]),
                                                    "score": float(clip_row[2]),
                                                    "chunk_start": chunk_start,
                                                    "chunk_end": chunk_end,
                                                    "touches_chunk_start_edge": float(clip_row[0]) <= chunk_first_end + 1e-6,
                                                    "touches_chunk_end_edge": float(clip_row[1]) >= chunk_last_start - 1e-6,
                                                    "from_cache": isinstance(cached, list),
                                                }
                                            )
                                        combined.extend(output)
                                        run_stats["ai_clipping_chunks_scanned"] += 1
                                        run_stats["ai_clipping_candidates"] += len(output)
                                        progress.update(
                                            ai_loop_task,
                                            completed=loop_index,
                                            description=(
                                                f"[magenta]AI loops: {loop_index}/{loop_count} "
                                                f"(chunk {chunk_index}/{total_chunks})"
                                            ),
                                        )
                                        refresh_activity_panel()

                                    raw_clip_groups.append(combined)
                                    progress.update(
                                        chunk_progress_task,
                                        completed=chunk_index,
                                        description=f"[magenta]Chunks: {chunk_index}/{total_chunks}",
                                    )

                                checkpoint["ai_scan"] = ai_scan_state
                                checkpoint["raw_clip_groups"] = raw_clip_groups
                                self._save_video_checkpoint(video, checkpoint)
                                if self.diagnostics.enabled:
                                    self.diagnostics.record_raw_candidates(
                                        video.name,
                                        candidates=raw_candidate_details,
                                        raw_clip_groups=raw_clip_groups,
                                    )
                                progress.update(chunk_progress_task, visible=False)
                                progress.update(ai_loop_task, visible=False)
                                progress.advance(activity_task)

                                self._assert_settings_unchanged(config_signature)
                                status["current_step"] = "merging_segments"
                                self._write_status(status)
                                progress.update(activity_task, description="[green]Activity: Assembling transcript anchors")
                                progress.refresh()
                                merged_clips = checkpoint.get("merged_clips")
                                merged_clip_details = checkpoint.get("merged_clip_details")
                                if not isinstance(merged_clips, list) or not isinstance(merged_clip_details, list):
                                    merged_clips, merged_clip_details = assemble_anchor_clips(
                                        transcript=transcript,
                                        candidates=raw_candidate_details,
                                        merge_distance_seconds=merge_distance,
                                        min_duration_seconds=min_duration_seconds,
                                    )
                                    checkpoint["merged_clips"] = merged_clips
                                    checkpoint["merged_clip_details"] = merged_clip_details
                                    self._save_video_checkpoint(video, checkpoint)
                                if self.diagnostics.enabled:
                                    self.diagnostics.record_merged_clips(
                                        video.name,
                                        merged_clips=merged_clips,
                                        merge_distance_seconds=merge_distance,
                                        details=merged_clip_details,
                                    )
                                run_stats["merged_candidates"] += len(merged_clips)
                                refresh_activity_panel()
                                progress.advance(activity_task)

                                self._assert_settings_unchanged(config_signature)
                                status["current_step"] = "filtering_candidates"
                                self._write_status(status)
                                progress.update(activity_task, description="[green]Activity: Filtering merged candidates")
                                progress.refresh()
                                final_clips = checkpoint.get("final_clips")
                                rejected_short = 0
                                if not isinstance(final_clips, list):
                                    final_clips, rejected_short = self._filter_min_duration(
                                        merged_clips,
                                        min_duration_seconds=min_duration_seconds,
                                    )
                                    checkpoint["final_clips"] = final_clips
                                    checkpoint["rejected_short"] = int(rejected_short)
                                    self._save_video_checkpoint(video, checkpoint)
                                else:
                                    rejected_short = int(checkpoint.get("rejected_short", 0))
                                if rejected_short > 0:
                                    self.logger.debug(
                                        "Removed %s merged candidates shorter than %.1f seconds.",
                                        rejected_short,
                                        min_duration_seconds,
                                    )
                                if self.diagnostics.enabled:
                                    self.diagnostics.record_final_clips(
                                        video.name,
                                        final_clips=final_clips,
                                        rejected_short=rejected_short,
                                        min_duration_seconds=min_duration_seconds,
                                    )
                                progress.advance(activity_task)

                                self._assert_settings_unchanged(config_signature)
                                status["current_step"] = "extracting_clips"
                                self._write_status(status)
                                extraction_state = checkpoint.get("extraction")
                                if not isinstance(extraction_state, dict):
                                    extraction_state = {"completed_indices": []}
                                completed_raw = extraction_state.get("completed_indices", [])
                                completed_indices: Set[int] = {
                                    int(value)
                                    for value in completed_raw
                                    if str(value).strip().isdigit()
                                }

                                if not final_clips:
                                    progress.advance(activity_task)
                                    progress.update(activity_task, description="[green]Activity: Archiving source")
                                    progress.refresh()
                                    self._archive_video(video)
                                    self._apply_temp_cleanup_policy()
                                    progress.advance(activity_task)
                                    self._clear_video_checkpoint(
                                        video,
                                        fingerprint=str(checkpoint.get("video_fingerprint", "")),
                                    )
                                    processed_video = True
                                    break

                                all_indices = list(range(1, len(final_clips) + 1))
                                remaining_indices = [clip_idx for clip_idx in all_indices if clip_idx not in completed_indices]
                                remaining_clips = [final_clips[clip_idx - 1] for clip_idx in remaining_indices]

                                progress.update(activity_task, description="[green]Activity: Extracting clips")
                                progress.refresh()

                                def on_clip_done(clip_index: int, output_path: str, created: bool) -> None:
                                    self._assert_settings_unchanged(config_signature)
                                    completed_indices.add(int(clip_index))
                                    extraction_state["completed_indices"] = sorted(completed_indices)
                                    extraction_state["total_clips"] = len(final_clips)
                                    checkpoint["extraction"] = extraction_state
                                    self._save_video_checkpoint(video, checkpoint)
                                    if self.diagnostics.enabled:
                                        self.diagnostics.record_extracted_clip(
                                            video.name,
                                            clip_index=clip_index,
                                            output_path=output_path,
                                            created=created,
                                            requested_clip=final_clips[clip_index - 1],
                                        )
                                    if created:
                                        run_stats["clips_extracted"] += 1
                                        refresh_activity_panel()

                                created = 0
                                if remaining_clips:
                                    created = extract_clips(
                                        clips=remaining_clips,
                                        source_video=video,
                                        output_dir=self.output_dir,
                                        logger=self.logger,
                                        progress_interval=progress_interval,
                                        clip_name_indices=remaining_indices,
                                        skip_existing=True,
                                        on_clip_done=on_clip_done,
                                    )
                                self.logger.info(
                                    "Extracted %s clips from %s (candidates=%s after_filter=%s)",
                                    created,
                                    video.name,
                                    len(merged_clips),
                                    len(final_clips),
                                )
                                progress.advance(activity_task)
                                progress.update(activity_task, description="[green]Activity: Archiving source")
                                progress.refresh()
                                self._archive_video(video)
                                self._apply_temp_cleanup_policy()
                                progress.advance(activity_task)
                                self._clear_video_checkpoint(
                                    video,
                                    fingerprint=str(checkpoint.get("video_fingerprint", "")),
                                )
                                processed_video = True
                            except _SettingsChangedError:
                                self.logger.warning(
                                    "Settings changed while processing %s. Restarting this video.",
                                    video.name,
                                )
                                self.console.print(
                                    "[yellow]Settings changed during run. Restarting current video.[/yellow]"
                                )
                                self._clear_video_checkpoint(video)
                                config_signature = self._current_resume_signature()
                                ai_pipeline = self._build_ai_pipeline()
                                continue
                            except Exception as exc:
                                self.logger.exception("Error processing %s: %s", video.name, exc)
                                self.console.print(f"[red]Failed processing {video.name}: {exc}[/red]")
                                processed_video = True

                        status["progress_percent"] = round((index / len(videos)) * 100, 1)
                        self._write_status(status)
                        progress.update(chunk_progress_task, visible=False)
                        progress.update(ai_loop_task, visible=False)
                        progress.advance(videos_task)
                        progress.update(
                            videos_task,
                            description=f"[cyan]Videos: {index}/{len(videos)}",
                        )
    
        finally:
            self._offload_runtime_resources(whisper_model=whisper_model)

        status["current_step"] = "completed"
        status["current_video"] = None
        status["progress_percent"] = 100.0
        self._write_status(status)
        summary = Table(show_header=False, box=None, pad_edge=False)
        summary.add_column("Key", style="cyan", no_wrap=True)
        summary.add_column("Value", style="white")
        summary.add_row("AI clipping base chunks", str(run_stats["ai_clipping_base_chunks"]))
        summary.add_row("AI clipping bridge chunks", str(run_stats["ai_clipping_bridge_chunks"]))
        summary.add_row("AI clipping chunks scanned", str(run_stats["ai_clipping_chunks_scanned"]))
        summary.add_row("AI clips found", str(run_stats["ai_clipping_candidates"]))
        summary.add_row("Candidates after merge", str(run_stats["merged_candidates"]))
        summary.add_row("Clips extracted", str(run_stats["clips_extracted"]))
        summary.add_row("Output folder", str(self.output_dir))
        summary.add_row("Run ID", run_id)
        if self.diagnostics.enabled:
            self.diagnostics.record_run_summary(
                {
                    "run_id": run_id,
                    "run_stats": dict(run_stats),
                    "output_dir": str(self.output_dir),
                }
            )
        self.console.print(Panel(summary, title="Run Summary", border_style="green"))
        self.console.print("[bold green]Engine run completed.[/bold green]")
        self.logger.info("Clipping engine run finished (run_id=%s)", run_id)
