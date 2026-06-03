from __future__ import annotations

import copy
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

from rich.columns import Columns
from rich.console import Console
from rich.console import Group
from rich.panel import Panel
from rich.prompt import Confirm, FloatPrompt, IntPrompt, Prompt
from rich.table import Table

from core.engine import ClippingEngine
from core.yt_handler import YTHandler
from ui.components import edit_text_in_editor, hard_clear, show_header, warning_panel
from ui.setup_wizard import SetupWizard
from utils.model_selector import (
    POLICY_MODEL,
    build_runtime_token_plan,
    ensure_ollama_running,
    fetch_local_models,
    is_required_thinking_model,
    model_exists_locally,
    prompt_aware_chunk_cap,
    pull_model,
    supports_think_low,
)
from utils.validators import config_paths, load_optional_profile, save_json_file, validate_config


class Dashboard:
    """Main dashboard shown after setup is complete."""

    _VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
    _YOUTUBE_WATCH_URL_PATTERN = re.compile(r"^https://www\.youtube\.com/watch\?v=[A-Za-z0-9_-]{11}$")

    def __init__(self, base_dir: Path, config: Dict[str, Any], console: Console, logger: logging.Logger) -> None:
        self.base_dir = base_dir
        self.config = config
        self.console = console
        self.logger = logger
        self.yt_handler = YTHandler(base_dir=base_dir, logger=logger)
        self._paths = config_paths(base_dir)

    def _save_config(self) -> None:
        save_json_file(self._paths["config_file"], self.config)

    def _recalculate_runtime_tokens(self, announce: bool = False) -> None:
        runtime_cfg = self.config.setdefault("runtime", {})
        clipping_cfg = self.config.setdefault("clipping", {})
        ollama_cfg = self.config.setdefault("ollama", {})

        hardware = load_optional_profile(self._paths["hardware_file"])
        intensity = str(runtime_cfg.get("setup_intensity", "balanced")).lower().strip() or "balanced"
        context_window = int(ollama_cfg.get("context_window", runtime_cfg.get("total_context_tokens", 8192)))
        max_output_tokens = int(ollama_cfg.get("max_output_tokens", 900))

        token_plan = build_runtime_token_plan(
            model_name=str(ollama_cfg.get("model", "")),
            context_window=context_window,
            max_output_tokens=max_output_tokens,
            hardware_profile=hardware,
            intensity=intensity,
        )
        runtime_cfg["total_context_tokens"] = int(token_plan["total_context_tokens"])
        runtime_cfg["max_chunk_tokens"] = int(token_plan["max_chunk_tokens"])

        prompt_plan = prompt_aware_chunk_cap(
            configured_chunk_tokens=int(runtime_cfg["max_chunk_tokens"]),
            total_context_tokens=int(runtime_cfg["total_context_tokens"]),
            max_output_tokens=int(max_output_tokens),
            user_query=str(clipping_cfg.get("user_query", "")),
            system_prompt=str(clipping_cfg.get("system_prompt", "")),
        )
        runtime_cfg["max_chunk_tokens"] = int(prompt_plan["effective_chunk_tokens"])

        # Backward-compat sync for old key still present in existing configs.
        clipping_cfg["rerun_temp_files"] = bool(runtime_cfg.get("enable_temp_run_save", True))

        if announce:
            self.console.print(
                (
                    "[green]Token settings updated:[/green] "
                    f"context={runtime_cfg['total_context_tokens']}, "
                    f"max_chunk={runtime_cfg['max_chunk_tokens']} "
                    f"(prompt_tokens~{prompt_plan['prompt_tokens']})."
                )
            )

    @classmethod
    def _parse_video_link_input(cls, raw_value: str) -> List[str]:
        return [item.strip() for item in re.split(r"[\s,]+", str(raw_value).strip()) if item.strip()]

    @classmethod
    def _normalize_video_links(cls, links: List[str]) -> tuple[List[str], List[str]]:
        normalized_links: List[str] = []
        invalid_links: List[str] = []
        seen: set[str] = set()

        for link in links:
            raw_link = str(link).strip()
            if not raw_link:
                continue
            normalized = YTHandler.normalize_video_url(raw_link)
            if not cls._YOUTUBE_WATCH_URL_PATTERN.fullmatch(normalized):
                invalid_links.append(raw_link)
                continue
            if normalized in seen:
                continue
            normalized_links.append(normalized)
            seen.add(normalized)
        return normalized_links, invalid_links

    def _set_youtube_queue(self, youtube_links: List[str]) -> bool:
        snapshot = copy.deepcopy(self.config)
        self.config["clipping"]["youtube_links"] = list(youtube_links)
        errors = validate_config(self.config)
        if errors:
            self.config = snapshot
            self.console.print(f"[red]Queue update rejected: {errors[0]}[/red]")
            return False
        self._save_config()
        return True

    def _resolve_path(self, configured_path: str) -> Path:
        path = Path(str(configured_path))
        if path.is_absolute():
            return path
        return (self.base_dir / path).resolve()

    def _count_waiting_input_videos(self) -> int:
        input_dir = self._resolve_path(str(self.config["paths"].get("input_dir", "input")))
        if not input_dir.exists():
            return 0
        count = 0
        for file_path in input_dir.iterdir():
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in self._VIDEO_EXTENSIONS:
                continue
            if file_path.name.endswith(".part"):
                continue
            count += 1
        return count

    def _count_output_clips(self) -> int:
        output_dir = self._resolve_path(str(self.config["paths"].get("output_dir", "output")))
        if not output_dir.exists():
            return 0
        return sum(1 for file_path in output_dir.glob("*.mp4") if file_path.is_file())

    def _status_panel(self) -> Panel:
        hardware = load_optional_profile(self._paths["hardware_file"])
        model_profile = load_optional_profile(self._paths["model_file"])
        waiting_videos = self._count_waiting_input_videos()
        output_clips = self._count_output_clips()

        pipeline_table = Table(show_header=False, box=None, pad_edge=False)
        pipeline_table.add_column("Key", style="cyan", no_wrap=True)
        pipeline_table.add_column("Value", style="white")
        pipeline_table.add_row("Model", str(self.config["ollama"]["model"]))
        pipeline_table.add_row("Whisper", str(self.config["transcription"]["model"]))
        pipeline_table.add_row("Intensity", str(self.config["runtime"].get("setup_intensity", "unknown")))
        pipeline_table.add_row(
            "Bridge Chunks",
            "Enabled" if bool(self.config["runtime"].get("enable_bridge_chunks", True)) else "Disabled",
        )
        bridge_chunks_enabled = bool(self.config["runtime"].get("enable_bridge_chunks", True))
        if bridge_chunks_enabled:
            pipeline_table.add_row(
                "Bridge Edge Segments",
                str(int(self.config["runtime"].get("bridge_chunk_edge_segments", 4))),
            )
        pipeline_table.add_row("AI Loops", str(int(self.config["clipping"].get("ai_loops", 1))))
        pipeline_table.add_row(
            "Assembly Gap",
            f"{int(self.config['clipping'].get('merge_distance_seconds', 20))}s",
        )

        queue_table = Table(show_header=False, box=None, pad_edge=False)
        queue_table.add_column("Key", style="cyan", no_wrap=True)
        queue_table.add_column("Value", style="white")
        queue_table.add_row("Queued Links", str(len(self.config["clipping"].get("youtube_links", []))))
        queue_table.add_row(
            "YouTube Downloading",
            "Enabled" if bool(self.config["clipping"].get("enable_youtube_downloads", True)) else "Disabled",
        )
        queue_table.add_row("Input Videos Waiting", str(waiting_videos))
        queue_table.add_row("Output Clips", str(output_clips))
        queue_table.add_row("Output Dir", str(self.config["paths"]["output_dir"]))
        queue_table.add_row("Ollama URL", str(self.config["ollama"]["url"]))
        queue_table.add_row(
            "Temp Run Save",
            "Enabled" if bool(self.config["runtime"].get("enable_temp_run_save", True)) else "Disabled",
        )
        cleanup = self.config.get("maintenance", {}).get("temp_cleanup", {})
        cleanup_mode = str(cleanup.get("mode", "never"))
        queue_table.add_row("Temp Cleanup", cleanup_mode)
        queue_table.add_row(
            "Log Mode",
            "DEV (DEBUG)" if self.config["app"].get("dev_mode") else "PROD (INFO)",
        )
        if model_profile.get("selection_source"):
            queue_table.add_row("Model Source", str(model_profile["selection_source"]))

        hardware_table = Table(show_header=False, box=None, pad_edge=False)
        hardware_table.add_column("Key", style="cyan", no_wrap=True)
        hardware_table.add_column("Value", style="white")
        hardware_table.add_row("GPU", str(hardware.get("gpu_model", "Unknown")))
        hardware_table.add_row(
            "GPU Acceleration",
            str(hardware.get("gpu_acceleration_available", "Unknown")),
        )
        hardware_table.add_row("CPU", str(hardware.get("cpu_model", "Unknown")))
        hardware_table.add_row("RAM", f"{hardware.get('ram_gb', '?')} GB")
        hardware_table.add_row("VRAM", f"{hardware.get('gpu_vram_gb', '?')} GB")

        layout = Group(
            Columns(
                [
                    Panel(pipeline_table, title="Pipeline", border_style="bright_cyan"),
                    Panel(queue_table, title="Queue + Paths", border_style="green"),
                ],
                equal=True,
                expand=True,
            ),
            Panel(hardware_table, title="Hardware", border_style="magenta"),
        )
        return Panel(layout, title="System Status", border_style="cyan")

    def _render_dashboard(self) -> None:
        hard_clear(self.console)
        show_header(self.console)
        self.console.print(self._status_panel())
        self.console.print(
            Panel(
                (
                    "[bold cyan]1.[/bold cyan] Launch Clipping Engine\n"
                    "[bold cyan]2.[/bold cyan] Change Settings\n"
                    "[bold cyan]3.[/bold cyan] YouTube Settings\n"
                    "[bold cyan]4.[/bold cyan] Re-run Setup Wizard\n"
                    "[bold cyan]5.[/bold cyan] Edit System Prompt\n"
                    "[bold cyan]6.[/bold cyan] Exit\n"
                    "[bold cyan]7.[/bold cyan] Info\n\n"
                    "[bold cyan]L.[/bold cyan] View Logs"
                ),
                title="Main Dashboard",
                border_style="green",
            )
        )

    def _fetch_youtube_links(self) -> None:
        channels = list(self.config["clipping"].get("channels", []))
        sanitized_channels = []
        for channel in channels:
            normalized = YTHandler.normalize_channel_url(str(channel))
            if normalized:
                sanitized_channels.append(normalized)
        channels = sorted(set(sanitized_channels))
        self.config["clipping"]["channels"] = channels
        self._save_config()

        if not channels:
            self.console.print(
                "[yellow]No channels configured yet. Add them in Settings > Channels.[/yellow]"
            )
            return

        hours_limit = int(self.config["clipping"].get("channels_hours_limit", 24))
        with self.console.status("[bold cyan]Fetching recent channel uploads...", spinner="dots"):
            links = self.yt_handler.fetch_recent_links(channels=channels, hours_limit=hours_limit, playlistend=15)

        current = list(self.config["clipping"].get("youtube_links", []))
        before_count = len(current)
        for link in links:
            if link not in current:
                current.append(link)
        self.config["clipping"]["youtube_links"] = current
        self._save_config()

        added = len(current) - before_count
        self.console.print(f"[green]Fetched {len(links)} links, added {added} new links to queue.[/green]")

    def _youtube_settings(self) -> None:
        while True:
            hard_clear(self.console)
            show_header(self.console)

            youtube_links = list(self.config["clipping"].get("youtube_links", []))
            channels = list(self.config["clipping"].get("channels", []))
            hours_limit = int(self.config["clipping"].get("channels_hours_limit", 24))
            enabled = bool(self.config["clipping"].get("enable_youtube_downloads", True))
            diagnostics = self.yt_handler.diagnostics()

            status_table = Table(title="YouTube Settings", show_lines=True)
            status_table.add_column("Setting", style="cyan", no_wrap=True)
            status_table.add_column("Value", style="white")
            status_table.add_row("Downloading", "Enabled" if enabled else "Disabled")
            status_table.add_row("Channels", str(len(channels)))
            status_table.add_row("Fetch Hours", str(hours_limit))
            status_table.add_row("Queued Links", str(len(youtube_links)))
            status_table.add_row(
                "Cookie Source",
                str(diagnostics.get("cookie_file") or diagnostics.get("browser_cookie_source") or "None"),
            )
            js_runtimes = diagnostics.get("js_runtimes") or []
            status_table.add_row("JS Runtime", ", ".join(js_runtimes) if js_runtimes else "None")
            self.console.print(status_table)

            table = Table(title="YouTube Queue", show_lines=True)
            table.add_column("#", style="cyan", no_wrap=True)
            table.add_column("Video URL", style="white")
            table.add_column("Status", style="white")
            if youtube_links:
                for index, link in enumerate(youtube_links, start=1):
                    status = "Downloaded" if self.yt_handler.is_in_history(link) else "Queued"
                    table.add_row(str(index), link, status)
            else:
                table.add_row("-", "No queued links", "-")
            self.console.print(table)

            self.console.print(
                Panel(
                    (
                        "[bold cyan]1.[/bold cyan] Toggle YouTube downloading\n"
                        "[bold cyan]2.[/bold cyan] Manage channels\n"
                        "[bold cyan]3.[/bold cyan] Set channel fetch hours\n"
                        "[bold cyan]4.[/bold cyan] Fetch recent links from channels\n"
                        "[bold cyan]5.[/bold cyan] Add links manually\n"
                        "[bold cyan]6.[/bold cyan] Remove queued link by index\n"
                        "[bold cyan]7.[/bold cyan] Replace entire queue\n"
                        "[bold cyan]8.[/bold cyan] Remove already-downloaded links\n"
                        "[bold cyan]9.[/bold cyan] Clear queue\n"
                        "[bold cyan]10.[/bold cyan] Run YouTube diagnostics probe\n"
                        "[bold cyan]0.[/bold cyan] Back"
                    ),
                    title="YouTube Actions",
                    border_style="cyan",
                )
            )

            action = Prompt.ask("Choose an action", default="0").strip()
            if action == "0":
                return

            if action == "1":
                self.config["clipping"]["enable_youtube_downloads"] = not enabled
                self._save_config()
                state = "enabled" if self.config["clipping"]["enable_youtube_downloads"] else "disabled"
                self.console.print(f"[green]YouTube downloading {state}.[/green]")
                input("Press Enter to continue...")
                continue

            if action == "2":
                self._manage_channels()
                continue

            if action == "3":
                self.config["clipping"]["channels_hours_limit"] = IntPrompt.ask(
                    "Fetch uploads from last X hours",
                    default=hours_limit,
                )
                errors = validate_config(self.config)
                if errors:
                    self.console.print(f"[red]Update rejected: {errors[0]}[/red]")
                else:
                    self._save_config()
                    self.console.print("[green]Fetch hours updated.[/green]")
                input("Press Enter to continue...")
                continue

            if action == "4":
                self._fetch_youtube_links()
                input("Press Enter to continue...")
                continue

            if action == "5":
                raw = Prompt.ask("Paste YouTube links or 11-char IDs (comma/space separated)").strip()
                parsed = self._parse_video_link_input(raw)
                normalized, invalid = self._normalize_video_links(parsed)
                if not normalized:
                    self.console.print("[yellow]No valid YouTube links were provided.[/yellow]")
                    if invalid:
                        preview = ", ".join(invalid[:3])
                        suffix = "" if len(invalid) <= 3 else f" (+{len(invalid) - 3} more)"
                        self.console.print(f"[yellow]Ignored: {preview}{suffix}[/yellow]")
                    input("Press Enter to continue...")
                    continue

                merged = list(youtube_links)
                before_count = len(merged)
                for link in normalized:
                    if link not in merged:
                        merged.append(link)

                if self._set_youtube_queue(merged):
                    added = len(merged) - before_count
                    self.console.print(
                        f"[green]Added {added} link(s). Queue now has {len(merged)} link(s).[/green]"
                    )
                if invalid:
                    preview = ", ".join(invalid[:3])
                    suffix = "" if len(invalid) <= 3 else f" (+{len(invalid) - 3} more)"
                    self.console.print(f"[yellow]Ignored invalid entries: {preview}{suffix}[/yellow]")
                input("Press Enter to continue...")
                continue

            if action == "6":
                if not youtube_links:
                    input("No queued links to remove. Press Enter...")
                    continue
                idx = IntPrompt.ask("Index to remove", default=1)
                if not (1 <= idx <= len(youtube_links)):
                    input("Invalid index. Press Enter...")
                    continue
                removed = youtube_links[idx - 1]
                del youtube_links[idx - 1]
                if self._set_youtube_queue(youtube_links):
                    self.console.print(f"[green]Removed: {removed}[/green]")
                input("Press Enter to continue...")
                continue

            if action == "7":
                raw = Prompt.ask("Paste replacement YouTube links or IDs (comma/space separated)").strip()
                parsed = self._parse_video_link_input(raw)
                normalized, invalid = self._normalize_video_links(parsed)
                if not normalized and not Confirm.ask("No valid links found. Replace queue with empty list?", default=False):
                    continue
                if self._set_youtube_queue(normalized):
                    self.console.print(f"[green]Queue replaced. New size: {len(normalized)} link(s).[/green]")
                if invalid:
                    preview = ", ".join(invalid[:3])
                    suffix = "" if len(invalid) <= 3 else f" (+{len(invalid) - 3} more)"
                    self.console.print(f"[yellow]Ignored invalid entries: {preview}{suffix}[/yellow]")
                input("Press Enter to continue...")
                continue

            if action == "8":
                if not youtube_links:
                    input("Queue is already empty. Press Enter...")
                    continue
                remaining = [link for link in youtube_links if not self.yt_handler.is_in_history(link)]
                removed_count = len(youtube_links) - len(remaining)
                if removed_count == 0:
                    self.console.print("[yellow]No queued links were found in download history.[/yellow]")
                    input("Press Enter to continue...")
                    continue
                if self._set_youtube_queue(remaining):
                    self.console.print(f"[green]Removed {removed_count} already-downloaded link(s).[/green]")
                input("Press Enter to continue...")
                continue

            if action == "9":
                if not youtube_links:
                    input("Queue is already empty. Press Enter...")
                    continue
                if not Confirm.ask("Clear all queued YouTube links?", default=False):
                    continue
                if self._set_youtube_queue([]):
                    self.console.print("[green]Queue cleared.[/green]")
                input("Press Enter to continue...")
                continue

            if action == "10":
                ok, message = self.yt_handler.probe_video_access()
                if ok:
                    self.console.print(f"[green]{message}[/green]")
                else:
                    self.console.print(
                        Panel(
                            f"[yellow]{message}[/yellow]",
                            title="YouTube Probe Warning",
                            border_style="yellow",
                        )
                    )
                input("Press Enter to continue...")
                continue

            input("Invalid option. Press Enter...")

    def _manage_channels(self) -> None:
        while True:
            hard_clear(self.console)
            show_header(self.console)
            channels = list(self.config["clipping"].get("channels", []))
            table = Table(title="Channel Manager", show_lines=True)
            table.add_column("#", style="cyan", no_wrap=True)
            table.add_column("Channel URL", style="white")
            if channels:
                for index, channel in enumerate(channels, start=1):
                    table.add_row(str(index), channel)
            else:
                table.add_row("-", "No channels configured")
            self.console.print(table)
            self.console.print(
                Panel(
                    (
                        "[bold cyan]1.[/bold cyan] Add channels\n"
                        "[bold cyan]2.[/bold cyan] Remove channel by index\n"
                        "[bold cyan]3.[/bold cyan] Replace all channels\n"
                        "[bold cyan]4.[/bold cyan] Clear all channels\n"
                        "[bold cyan]0.[/bold cyan] Back"
                    ),
                    title="Channels",
                    border_style="cyan",
                )
            )
            action = Prompt.ask("Choose an action", default="0").strip()
            if action == "0":
                return

            if action == "1":
                raw = Prompt.ask("Paste channel URLs or handles, comma-separated").strip()
                parsed = [item.strip() for item in raw.split(",") if item.strip()]
                normalized = [YTHandler.normalize_channel_url(item) for item in parsed]
                normalized = [item for item in normalized if item]
                merged = sorted(set(channels + normalized))
                self.config["clipping"]["channels"] = merged
            elif action == "2":
                if not channels:
                    input("No channels to remove. Press Enter...")
                    continue
                idx = IntPrompt.ask("Index to remove", default=1)
                if 1 <= idx <= len(channels):
                    del channels[idx - 1]
                    self.config["clipping"]["channels"] = channels
                else:
                    input("Invalid index. Press Enter...")
                    continue
            elif action == "3":
                raw = Prompt.ask("Paste replacement channel URLs/handles, comma-separated").strip()
                parsed = [item.strip() for item in raw.split(",") if item.strip()]
                normalized = [YTHandler.normalize_channel_url(item) for item in parsed]
                normalized = [item for item in normalized if item]
                self.config["clipping"]["channels"] = sorted(set(normalized))
            elif action == "4":
                if Confirm.ask("Clear all configured channels?", default=False):
                    self.config["clipping"]["channels"] = []
            else:
                input("Invalid option. Press Enter...")
                continue

            errors = validate_config(self.config)
            if errors:
                self.console.print(f"[red]Channel update rejected: {errors[0]}[/red]")
                input("Press Enter to continue...")
                continue

            self._save_config()
            self.console.print("[green]Channels updated.[/green]")
            input("Press Enter to continue...")

    def _configure_bridge_chunks(self) -> bool:
        runtime_cfg = self.config.setdefault("runtime", {})
        hard_clear(self.console)
        show_header(self.console)
        enabled = bool(runtime_cfg.get("enable_bridge_chunks", True))
        edge_segments = int(runtime_cfg.get("bridge_chunk_edge_segments", 4))

        self.console.print(
            Panel(
                (
                    f"[bold cyan]Current:[/bold cyan] {'Enabled' if enabled else 'Disabled'}\n"
                    f"[bold cyan]Edge Segments:[/bold cyan] {edge_segments}\n\n"
                    "[bold cyan]1.[/bold cyan] Enable\n"
                    "[bold cyan]2.[/bold cyan] Disable\n"
                    "[bold cyan]3.[/bold cyan] Change value\n"
                    "[bold cyan]4.[/bold cyan] Back"
                ),
                title="Bridge Edge Segments",
                border_style="cyan",
            )
        )

        choice = Prompt.ask("Select an option", choices=["1", "2", "3", "4"], default="4")
        if choice == "4":
            return False
        if choice == "1":
            runtime_cfg["enable_bridge_chunks"] = True
            return True
        if choice == "2":
            runtime_cfg["enable_bridge_chunks"] = False
            return True

        runtime_cfg["bridge_chunk_edge_segments"] = IntPrompt.ask(
            "Bridge edge segments (1-20)",
            default=edge_segments,
        )
        return True

    def _configure_temp_run_save(self) -> bool:
        runtime_cfg = self.config.setdefault("runtime", {})
        clipping_cfg = self.config.setdefault("clipping", {})
        hard_clear(self.console)
        show_header(self.console)
        enabled = bool(runtime_cfg.get("enable_temp_run_save", clipping_cfg.get("rerun_temp_files", True)))

        self.console.print(
            Panel(
                (
                    f"[bold cyan]Current:[/bold cyan] {'Enabled' if enabled else 'Disabled'}\n\n"
                    "[bold cyan]1.[/bold cyan] Enable\n"
                    "[bold cyan]2.[/bold cyan] Disable\n"
                    "[bold cyan]3.[/bold cyan] Back"
                ),
                title="Temp Run Save",
                border_style="cyan",
            )
        )

        choice = Prompt.ask("Select an option", choices=["1", "2", "3"], default="3")
        if choice == "3":
            return False

        runtime_cfg["enable_temp_run_save"] = choice == "1"
        clipping_cfg["rerun_temp_files"] = runtime_cfg["enable_temp_run_save"]
        state = "enabled" if runtime_cfg["enable_temp_run_save"] else "disabled"
        self.console.print(f"[green]Temp run save {state}.[/green]")
        return True

    def _change_settings(self) -> None:
        while True:
            hard_clear(self.console)
            show_header(self.console)
            self.console.print(
                Panel(
                    (
                        "[bold cyan]1.[/bold cyan] Ollama Model\n"
                        "[bold cyan]2.[/bold cyan] Whisper Model\n"
                        "[bold cyan]3.[/bold cyan] User Query (Editor)\n"
                        "[bold cyan]4.[/bold cyan] Assembly Gap\n"
                        "[bold cyan]5.[/bold cyan] AI Loops\n"
                        "[bold cyan]6.[/bold cyan] Temperature\n"
                        "[bold cyan]7.[/bold cyan] Output Directory\n"
                        "[bold cyan]8.[/bold cyan] Performance Level\n"
                        "[bold cyan]9.[/bold cyan] Log Mode (Prod/Dev)\n"
                        "[bold cyan]10.[/bold cyan] Temp Cleanup Policy\n"
                        "[bold cyan]11.[/bold cyan] Chunk Overlap Segments\n"
                        "[bold cyan]12.[/bold cyan] Bridge Edge Segments\n"
                        "[bold cyan]13.[/bold cyan] Temp Run Save\n"
                        "[bold cyan]0.[/bold cyan] Back"
                    ),
                    title="Settings",
                    border_style="cyan",
                )
            )
            choice = Prompt.ask("Select an option", default="0")
            snapshot = copy.deepcopy(self.config)
            recalc_tokens = False

            if choice == "0":
                return
            if choice == "1":
                model_name = Prompt.ask("Model name", default=str(self.config["ollama"]["model"])).strip()
                if model_name:
                    if not is_required_thinking_model(model_name):
                        self.console.print(
                            "[red]Model policy: only thinking models with >=8b are allowed.[/red]"
                        )
                        continue
                    if model_name.lower() != POLICY_MODEL.lower():
                        self.console.print(
                            warning_panel(
                                (
                                    f"`{POLICY_MODEL}` is strongly recommended for this system.\n"
                                    f"`{model_name}` may provide significantly worse clipping reliability."
                                )
                            )
                        )
                        if not Confirm.ask("Switch to this non-recommended model anyway?", default=False):
                            continue
                    ollama_url = str(self.config["ollama"]["url"])
                    if ensure_ollama_running(ollama_url, self.logger):
                        local_models = fetch_local_models(ollama_url)
                        if not model_exists_locally(model_name, local_models):
                            self.console.print(f"[cyan]Pulling `{model_name}`...[/cyan]")
                            ok, output = pull_model(model_name, self.logger)
                            if not ok:
                                self.console.print(f"[red]Model pull failed: {output}[/red]")
                                continue
                        self.config["ollama"]["model"] = model_name
                        if model_name.lower() != POLICY_MODEL.lower():
                            self.console.print(
                                warning_panel(
                                    (
                                        f"`{POLICY_MODEL}` remains the recommended model.\n"
                                        f"`{model_name}` may perform significantly worse for this workflow."
                                    )
                                )
                            )
                        if not supports_think_low(model_name):
                            self.console.print(
                                warning_panel(
                                    (
                                        f"`{model_name}` does not support stable `think='low'` control in this stack. "
                                        "Expect slower runs and less stable clip quality."
                                    )
                                )
                            )
                    recalc_tokens = True
            elif choice == "2":
                self.config["transcription"]["model"] = Prompt.ask(
                    "Whisper model",
                    choices=["tiny", "base", "small", "medium", "large"],
                    default=str(self.config["transcription"]["model"]),
                )
            elif choice == "3":
                current_query = str(self.config["clipping"]["user_query"])
                self.console.print("[cyan]Opening editor for user query...[/cyan]")
                query = edit_text_in_editor(
                    current_query,
                    filename_hint="verbacut_user_query.txt",
                ).strip()
                if query:
                    self.config["clipping"]["user_query"] = query
                    recalc_tokens = True
            elif choice == "4":
                self.config["clipping"]["merge_distance_seconds"] = IntPrompt.ask(
                    "Assembly gap (seconds)",
                    default=int(self.config["clipping"]["merge_distance_seconds"]),
                )
            elif choice == "5":
                self.config["clipping"]["ai_loops"] = IntPrompt.ask(
                    "AI loops",
                    default=int(self.config["clipping"]["ai_loops"]),
                )
            elif choice == "6":
                self.config["ollama"]["temperature"] = FloatPrompt.ask(
                    "Temperature",
                    default=float(self.config["ollama"]["temperature"]),
                )
            elif choice == "7":
                output_dir = Prompt.ask(
                    "Output clips directory (relative or absolute)",
                    default=str(self.config["paths"]["output_dir"]),
                ).strip()
                if output_dir:
                    self.config["paths"]["output_dir"] = str(Path(output_dir.strip('"')))
            elif choice == "8":
                runtime_cfg = self.config.setdefault("runtime", {})
                runtime_cfg["setup_intensity"] = Prompt.ask(
                    "Performance level",
                    choices=["light", "balanced", "maximum"],
                    default=str(runtime_cfg.get("setup_intensity", "balanced")),
                )
                recalc_tokens = True
            elif choice == "9":
                dev_mode = Confirm.ask(
                    "Enable dev logging (DEBUG)?",
                    default=bool(self.config["app"].get("dev_mode", False)),
                )
                self.config["app"]["dev_mode"] = dev_mode
                self.config["app"]["log_level"] = "DEBUG" if dev_mode else "INFO"
            elif choice == "10":
                cleanup = self.config.setdefault("maintenance", {}).setdefault("temp_cleanup", {})
                mode = Prompt.ask(
                    "Temp cleanup mode",
                    choices=["never", "max_size_gb", "max_age_days"],
                    default=str(cleanup.get("mode", "never")),
                )
                cleanup["mode"] = mode
                if mode == "max_size_gb":
                    cleanup["max_size_gb"] = IntPrompt.ask(
                        "Delete oldest when temp exceeds (GB)",
                        default=int(cleanup.get("max_size_gb", 20)),
                    )
                elif mode == "max_age_days":
                    cleanup["max_age_days"] = IntPrompt.ask(
                        "Delete files older than (days)",
                        default=int(cleanup.get("max_age_days", 30)),
                    )
                else:
                    cleanup["max_size_gb"] = int(cleanup.get("max_size_gb", 20))
                    cleanup["max_age_days"] = int(cleanup.get("max_age_days", 30))
            elif choice == "11":
                runtime_cfg = self.config.setdefault("runtime", {})
                runtime_cfg["chunk_overlap_segments"] = IntPrompt.ask(
                    "Chunk overlap segments (0-20)",
                    default=int(runtime_cfg.get("chunk_overlap_segments", 3)),
                )
            elif choice == "12":
                changed = self._configure_bridge_chunks()
                if not changed:
                    continue
            elif choice == "13":
                changed = self._configure_temp_run_save()
                if not changed:
                    continue
            else:
                self.console.print("[yellow]Invalid option.[/yellow]")
                continue

            errors = validate_config(self.config)
            if errors:
                self.config = snapshot
                self.console.print(f"[red]Setting change rejected: {errors[0]}[/red]")
                continue

            if recalc_tokens:
                self._recalculate_runtime_tokens(announce=True)
                errors = validate_config(self.config)
                if errors:
                    self.config = snapshot
                    self.console.print(f"[red]Setting change rejected after token update: {errors[0]}[/red]")
                    continue

            self._save_config()
            self.console.print("[green]Settings saved.[/green]")

    def _edit_system_prompt(self) -> None:
        hard_clear(self.console)
        show_header(self.console)
        self.console.print(
            warning_panel(
                "Editing system instructions is not recommended unless you understand prompt engineering."
            )
        )
        current_prompt = str(self.config["clipping"]["system_prompt"])
        self.console.print(Panel(current_prompt, title="Current System Prompt", border_style="cyan"))
        self.console.print("[cyan]Opening external editor for full prompt navigation/editing...[/cyan]")
        prompt_text = edit_text_in_editor(
            current_prompt,
            filename_hint="verbacut_system_prompt.txt",
        ).strip()
        if not prompt_text:
            self.console.print("[yellow]System prompt unchanged.[/yellow]")
            return

        self.config["clipping"]["system_prompt"] = prompt_text
        errors = validate_config(self.config)
        if errors:
            self.console.print(f"[red]Prompt update rejected: {errors[0]}[/red]")
            return
        self._recalculate_runtime_tokens(announce=True)
        errors = validate_config(self.config)
        if errors:
            self.console.print(f"[red]Prompt update rejected after token update: {errors[0]}[/red]")
            return
        self._save_config()
        self.console.print("[green]System prompt updated.[/green]")

    def _view_logs(self) -> None:
        hard_clear(self.console)
        show_header(self.console)
        logs_dir = self.base_dir / str(self.config["paths"].get("logs_dir", "logs"))
        log_path = logs_dir / "app.log"
        if not log_path.exists():
            self.console.print("[yellow]No log file found yet at logs/app.log.[/yellow]")
            return

        level_choice = Prompt.ask(
            "Filter logs",
            choices=["all", "warning", "error"],
            default="warning",
        )
        max_lines = IntPrompt.ask("How many latest lines to show", default=80, show_default=True)

        with log_path.open("r", encoding="utf-8", errors="replace") as handle:
            raw_lines = [line.rstrip() for line in handle.readlines() if line.strip()]

        selected = raw_lines[-max(1, max_lines) :]
        entries: List[List[str]] = []
        for line in selected:
            parts = [part.strip() for part in line.split("|", 3)]
            if len(parts) >= 4:
                timestamp, level, source, message = parts[0], parts[1], parts[2], parts[3]
            else:
                timestamp, level, source, message = "", "INFO", "", line

            upper = level.upper()
            if level_choice == "warning" and "WARNING" not in upper:
                continue
            if level_choice == "error" and "ERROR" not in upper:
                continue
            entries.append([timestamp, upper, source, message])

        if not entries:
            self.console.print(f"[yellow]No {level_choice.upper()} log lines found in selection.[/yellow]")
            return

        table = Table(title=f"Log Viewer ({level_choice.upper()})", show_lines=True)
        table.add_column("Time", style="cyan", no_wrap=True)
        table.add_column("Level", no_wrap=True)
        table.add_column("Source", style="white", no_wrap=True)
        table.add_column("Message", style="white")

        for timestamp, level, source, message in entries:
            if "ERROR" in level:
                level_style = "[bold red]ERROR[/bold red]"
            elif "WARNING" in level:
                level_style = "[bold yellow]WARNING[/bold yellow]"
            elif "DEBUG" in level:
                level_style = "[magenta]DEBUG[/magenta]"
            else:
                level_style = "[green]INFO[/green]"
            table.add_row(timestamp, level_style, source, message)

        self.console.print(table)

    def _show_info(self) -> None:
        hard_clear(self.console)
        show_header(self.console)

        overview = Panel(
            (
                "[bold cyan]System Overview[/bold cyan]\n"
                "VerbaCut is a local-first clipping pipeline that downloads videos, transcribes speech, "
                "finds relevant clips with a reasoning model, and exports final shorts."
            ),
            title="Info",
            border_style="cyan",
        )
        self.console.print(overview)

        stages = Table(title="Pipeline Stages", show_lines=True)
        stages.add_column("Stage", style="cyan", no_wrap=True)
        stages.add_column("What It Does", style="white")
        stages.add_row("YouTube Download", "Downloads queued links into input/ using robust format fallback.")
        stages.add_row("Transcription", "Whisper converts speech to timestamped transcript segments.")
        stages.add_row("Chunking", "Transcript is split into token-safe chunks for stable model inference.")
        stages.add_row(
            "AI Clipping",
            "AI clipping scans each chunk (and loop) to find broad candidate moments scored by relevance.",
        )
        stages.add_row("Merge", "Nearby AI clipping candidates are merged into cleaner timeline windows.")
        stages.add_row("Filtering", "Merged candidates are filtered by duration constraints from the user query.")
        stages.add_row("Extraction", "Filtered clips are rendered to output/ as mp4 files.")
        stages.add_row("Archive/Cleanup", "Processed source videos move to temp/ with optional cleanup policy.")
        self.console.print(stages)

        settings = Table(title="Key Settings", show_lines=True)
        settings.add_column("Setting", style="cyan", no_wrap=True)
        settings.add_column("Meaning", style="white")
        settings.add_row(
            "AI Loops",
            "How many AI clipping scans run per chunk. Increasing this can recover more candidates but increases runtime and total model calls.",
        )
        settings.add_row(
            "Assembly Gap",
            "Soft upper bound used when grouping nearby AI-selected anchors. Higher values can widen continuous clips, but the system now still requires contiguous speech instead of blindly merging by time gap.",
        )
        settings.add_row(
            "Temperature",
            "Model randomness. Lower values improve strict JSON/output stability; higher values can increase variety but also parsing risk.",
        )
        settings.add_row(
            "Whisper Model",
            "Speech recognition size/speed tradeoff (tiny -> large). Larger models improve transcript quality but increase transcription time and VRAM/RAM usage.",
        )
        settings.add_row(
            "Chunk Overlap Segments",
            "How many transcript segments are repeated between adjacent chunks. Higher overlap improves continuity near chunk boundaries but creates more total chunk work.",
        )
        settings.add_row(
            "Bridge Edge Segments",
            "Extra synthetic bridge chunks from neighboring chunk edges. Higher values improve detection of long ideas crossing chunk borders but adds extra AI scans.",
        )
        settings.add_row(
            "Enable Bridge Chunks",
            "On: add transition chunks between neighboring base chunks (better boundary recall, slower). Off: scan only base chunks (faster).",
        )
        settings.add_row("AI Clipping", "Broad discovery stage for potential moments.")
        settings.add_row("Duration Constraints", "Requests like 'at least 30 seconds' are enforced in final clip filtering.")
        settings.add_row("Setup Guidance", "Setup wizard now explains each setting and its practical effect before you choose values.")
        settings.add_row("Temp Cleanup", "Never, max size (GB), or max age (days) for temp archive files.")
        settings.add_row("Thinking Policy", "Only thinking-capable models with >=8b are allowed.")
        self.console.print(settings)

        self.console.print(
            Panel(
                (
                    "[bold cyan]Performance Notes[/bold cyan]\n"
                    "- AI clipping now uses chunk/loop progress indicators.\n"
                    "- Activity Panel shows live AI clipping and extraction counters while the engine runs.\n"
                    "- Transcription stage now stays as a clean single activity line.\n"
                    "- AI requests use low-thinking mode for speed when supported.\n"
                    "- Smaller chunk sizes and lower loop counts reduce run time."
                ),
                title="Tips",
                border_style="green",
            )
        )
        input("Press Enter to return to dashboard...")

    def _rerun_setup_wizard(self) -> None:
        wizard = SetupWizard(base_dir=self.base_dir, console=self.console, logger=self.logger)
        self.config = wizard.run()
        self._save_config()
        self.console.print("[green]Setup wizard completed and config refreshed.[/green]")

    def _launch_engine(self) -> None:
        engine = ClippingEngine(
            base_dir=self.base_dir,
            config=self.config,
            logger=self.logger,
            console=self.console,
        )
        engine.run()
        input("Press Enter to return to dashboard...")

    def run(self) -> None:
        while True:
            self._render_dashboard()
            choice = Prompt.ask("Choose an action", default="6").strip().lower()
            if choice == "1":
                self._launch_engine()
            elif choice == "2":
                self._change_settings()
            elif choice == "3":
                self._youtube_settings()
            elif choice == "4":
                self._rerun_setup_wizard()
                input("Press Enter to continue...")
            elif choice == "5":
                self._edit_system_prompt()
                input("Press Enter to continue...")
            elif choice == "6":
                self.console.print("[bold cyan]Exiting VerbaCut.[/bold cyan]")
                return
            elif choice == "7":
                self._show_info()
            elif choice == "l":
                self._view_logs()
                input("Press Enter to continue...")
            else:
                self.console.print("[yellow]Invalid option. Try again.[/yellow]")
                input("Press Enter to continue...")
