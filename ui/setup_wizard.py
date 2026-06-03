from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, FloatPrompt, IntPrompt, Prompt
from rich.table import Table

from ui.components import (
    edit_text_in_editor,
    hard_clear,
    show_header,
    startup_animation,
    success_panel,
    warning_panel,
)
from utils.hardware_detect import detect_hardware_profile
from utils.model_selector import (
    POLICY_MODEL,
    ModelRecommendation,
    build_token_plan,
    estimate_budget_gb,
    ensure_ollama_running,
    fetch_local_models,
    is_thinking_model,
    is_required_thinking_model,
    model_billions,
    model_exists_locally,
    prompt_aware_chunk_cap,
    pull_model,
    recommend_model,
    supports_think_low,
)
from utils.validators import (
    DEFAULT_SYSTEM_PROMPT,
    build_default_config,
    config_paths,
    load_optional_profile,
    normalize_ollama_url,
    validate_config,
    write_config_bundle,
)


class SetupWizard:
    """Interactive first-run setup experience."""

    def __init__(self, base_dir: Path, console: Console, logger: logging.Logger) -> None:
        self.base_dir = base_dir
        self.console = console
        self.logger = logger

    def _show_step(self, title: str, subtitle: str = "") -> None:
        hard_clear(self.console)
        show_header(self.console)
        self.console.rule(f"[bold cyan]{title}")
        if subtitle:
            self.console.print(f"[white]{subtitle}[/white]")

    def _show_setting_help(self, name: str, behavior: str, impact: str) -> None:
        self.console.print(
            Panel(
                (
                    f"[bold cyan]{name}[/bold cyan]\n"
                    f"[white]{behavior}[/white]\n\n"
                    f"[bold]Effect:[/bold] {impact}"
                ),
                title="Setup Guidance",
                border_style="magenta",
            )
        )

    def _render_hardware_table(self, profile: Dict[str, Any]) -> None:
        table = Table(title="Detected Hardware", show_lines=True)
        table.add_column("Component", style="cyan", no_wrap=True)
        table.add_column("Detected", style="white")
        table.add_row("CPU", str(profile.get("cpu_model", "Unknown")))
        table.add_row("RAM", f"{profile.get('ram_gb', 0)} GB")
        table.add_row("GPU", str(profile.get("gpu_model", "None")))
        table.add_row("VRAM", f"{profile.get('gpu_vram_gb', 0)} GB")
        table.add_row("GPU Acceleration", str(profile.get("gpu_acceleration_available", False)))
        table.add_row("Safe Model Size", f"~{profile.get('safe_model_size_gb', 0)} GB")
        table.add_row("Max Token Estimate", str(profile.get("max_tokens_estimate", 0)))
        table.add_row("Recommended Context", str(profile.get("recommended_context_window", 0)))
        self.console.print(table)

    def _render_local_models(self, local_models: List[Dict[str, Any]]) -> None:
        if not local_models:
            self.console.print("[yellow]No local Ollama models detected yet.[/yellow]")
            return

        table = Table(title="Local Ollama Models")
        table.add_column("Model", style="cyan")
        table.add_column("Size (GB)", justify="right")
        table.add_column("Thinking", justify="center")
        for model in local_models:
            size = model.get("size_gb")
            size_text = f"{size:.2f}" if isinstance(size, (int, float)) else "?"
            thinking = "Yes" if model.get("thinking") else "No"
            table.add_row(str(model.get("name", "")), size_text, thinking)
        self.console.print(table)

    def _read_system_prompt(self, current_prompt: str) -> str:
        self.console.print(
            warning_panel(
                "Editing system instructions is not recommended unless you understand prompt engineering."
            )
        )
        self.console.print(
            Panel(
                current_prompt,
                title="Current System Prompt",
                border_style="cyan",
            )
        )
        self.console.print(
            "[bold cyan]Opening external editor so you can navigate and edit the full prompt.[/bold cyan]"
        )
        edited = edit_text_in_editor(current_prompt, filename_hint="verbacut_system_prompt.txt")
        candidate = edited.strip()
        return candidate or current_prompt

    @staticmethod
    def _normalize_output_path(raw_path: str, default_path: str) -> str:
        cleaned = (raw_path or "").strip().strip('"')
        if not cleaned:
            return default_path
        return str(Path(cleaned))

    def run(self) -> Dict[str, Any]:
        self._show_step("Starting Setup Wizard")
        startup_animation(self.console)

        self._show_step("Hardware Detection")
        with self.console.status("[bold cyan]Detecting hardware profile...", spinner="dots"):
            hardware_profile = detect_hardware_profile().to_dict()
        self._render_hardware_table(hardware_profile)
        self._show_setting_help(
            name="System Intensity",
            behavior="Controls how aggressively token/context budgets are planned against your hardware profile.",
            impact="Light is safer and cooler, Balanced is default, Maximum increases context/token targets and can increase latency or memory pressure.",
        )
        intensity = Prompt.ask(
            "How heavy should the system run?",
            choices=["light", "balanced", "maximum"],
            default="balanced",
        )

        self._show_step("Ollama Connection")
        self._show_setting_help(
            name="Ollama URL",
            behavior="Defines which local Ollama server this tool will use for AI scanning requests.",
            impact="Wrong URL means model listing, pull, and inference will fail. Default localhost is correct for most users.",
        )
        ollama_url = normalize_ollama_url(
            Prompt.ask("Ollama URL", default="http://localhost:11434")
        )
        with self.console.status("[bold cyan]Checking Ollama service...", spinner="dots"):
            ollama_ready = ensure_ollama_running(ollama_url=ollama_url, logger=self.logger)
        if not ollama_ready:
            self.console.print(
                "[red]Ollama is not reachable. Install/start Ollama and rerun setup.[/red]"
            )
            raise RuntimeError("Ollama service unavailable during setup.")

        self._show_step("Model Selection")
        local_models = fetch_local_models(ollama_url=ollama_url)
        recommendation = recommend_model(
            hardware_profile=hardware_profile,
            intensity=intensity,
            local_models=local_models,
        )
        self.console.print(
            Panel(
                (
                    f"Recommended model: [bold]{recommendation.model_name}[/bold]\n"
                    f"Reason: {recommendation.reason}\n"
                    f"Estimated size: {recommendation.estimated_size_gb:.1f} GB\n"
                    f"Source: {recommendation.source}"
                ),
                title="Model Recommendation",
                border_style="cyan",
            )
        )
        budget_gb = estimate_budget_gb(hardware_profile=hardware_profile, intensity=intensity)
        if recommendation.estimated_size_gb > budget_gb:
            self.console.print(
                warning_panel(
                    (
                        f"Hardware warning: `{POLICY_MODEL}` is ~{recommendation.estimated_size_gb:.1f} GB but "
                        f"current safe budget is ~{budget_gb:.1f} GB.\n"
                        "You can still continue, but expect slower AI clipping latency and higher system load."
                    )
                )
            )
        self._render_local_models(local_models)
        self._show_setting_help(
            name="Model Choice",
            behavior="Selects the reasoning model used for AI clipping discovery.",
            impact="Better reasoning models improve clip relevance and JSON stability. Smaller/weaker models reduce quality and increase invalid output risk.",
        )

        while True:
            selected_model = Prompt.ask("Model to use", default=recommendation.model_name).strip()
            if not selected_model:
                selected_model = recommendation.model_name
            if not is_required_thinking_model(selected_model):
                self.console.print(
                    "[red]Model policy: only thinking models with >=8b are allowed.[/red]"
                )
                continue

            if selected_model.lower() != POLICY_MODEL.lower():
                self.console.print(
                    warning_panel(
                        (
                            f"`{POLICY_MODEL}` is the recommended model for this system.\n"
                            f"`{selected_model}` does not support `think='low'` control like `{POLICY_MODEL}`.\n"
                            "You can continue, but clipping quality/speed stability may be significantly worse."
                        )
                    )
                )
                if not Confirm.ask("Use this non-recommended model anyway?", default=False):
                    continue
            break

        if not model_exists_locally(selected_model, local_models):
            self.console.print(f"[cyan]Model not installed locally. Pulling `{selected_model}`...[/cyan]")
            with self.console.status("[bold cyan]Running ollama pull...", spinner="dots"):
                pulled_ok, pull_output = pull_model(model_name=selected_model, logger=self.logger)
            if not pulled_ok:
                self.logger.error("Model pull failed: %s", pull_output)
                raise RuntimeError(f"Failed to pull model `{selected_model}`: {pull_output}")
            self.console.print(f"[green]Model pull completed for `{selected_model}`.[/green]")

        local_models = fetch_local_models(ollama_url)
        selected_recommendation = recommendation
        if selected_model.lower() != recommendation.model_name.lower():
            selected_meta = next(
                (item for item in local_models if str(item.get("name", "")).lower() == selected_model.lower()),
                {},
            )
            selected_size = selected_meta.get("size_gb")
            if not isinstance(selected_size, (int, float)):
                selected_size = recommendation.estimated_size_gb
            selected_recommendation = ModelRecommendation(
                model_name=selected_model,
                thinking=is_thinking_model(selected_model),
                source="user-selected",
                estimated_size_gb=float(selected_size),
                reason="User selected a custom model.",
                context_window=recommendation.context_window,
                max_output_tokens=1000 if is_thinking_model(selected_model) else 450,
            )
        selected_b = model_billions(selected_model)
        if selected_b is not None and selected_b < 8.0:
            raise RuntimeError("Selected model does not meet minimum 8b thinking requirement.")
        if not supports_think_low(selected_model):
            self.console.print(
                warning_panel(
                    (
                        f"`{selected_model}` is configured, but only `{POLICY_MODEL}` has stable `think='low'` "
                        "control in this workflow. Expect slower inference and weaker consistency."
                    )
                )
            )

        self._show_step("Clipping Pipeline")
        self.console.print(
            Panel(
                "[bold cyan]Every setting below includes what it changes.[/bold cyan]\n"
                "Use defaults if unsure; tune only when you need specific behavior.",
                title="Pipeline Setup",
                border_style="cyan",
            )
        )
        config_template = build_default_config()
        existing_config = load_optional_profile(config_paths(self.base_dir)["config_file"])
        runtime_defaults = existing_config.get("runtime", {})
        default_output_dir = str(
            existing_config.get("paths", {}).get(
                "output_dir",
                config_template["paths"]["output_dir"],
            )
        )
        self._show_setting_help(
            name="Output Directory",
            behavior="Where extracted final clips are written.",
            impact="Changing this only affects exported clip destination; source processing and temp archive behavior are unchanged.",
        )
        output_dir = self._normalize_output_path(
            Prompt.ask(
                "Output clips directory (relative or absolute)",
                default=default_output_dir,
            ),
            default_output_dir,
        )
        self._show_setting_help(
            name="Whisper Model",
            behavior="Speech transcription size/speed tradeoff.",
            impact="Larger Whisper models can improve transcript accuracy and clip relevance, but increase processing time and hardware load.",
        )
        whisper_model = Prompt.ask(
            "Whisper transcription model",
            choices=["tiny", "base", "small", "medium", "large"],
            default="base",
        )
        self._show_setting_help(
            name="User Query",
            behavior="Primary clipping target instruction for content style/theme.",
            impact="This heavily drives what AI clipping searches for and which candidates get extracted.",
        )
        default_user_query = str(
            existing_config.get("clipping", {}).get(
                "user_query",
                config_template["clipping"]["user_query"],
            )
        ).strip() or config_template["clipping"]["user_query"]
        self.console.print("[bold cyan]Opening editor for user query...[/bold cyan]")
        user_query = edit_text_in_editor(
            default_user_query,
            filename_hint="verbacut_user_query.txt",
        ).strip()
        if not user_query:
            user_query = default_user_query
        self._show_setting_help(
            name="Assembly Gap",
            behavior="Soft upper bound for grouping nearby AI anchor moments before transcript-boundary expansion.",
            impact="Higher values allow slightly wider continuous passages when speech stays contiguous. Lower values keep clips stricter and reduce over-joining.",
        )
        merge_distance_seconds = IntPrompt.ask("Assembly gap in seconds", default=20, show_default=True)
        self._show_setting_help(
            name="AI Loops",
            behavior="How many AI clipping loops run per chunk.",
            impact="More loops may improve recall but increase total runtime and model calls.",
        )
        ai_loops = IntPrompt.ask("AI scan loops per chunk", default=2, show_default=True)
        self._show_setting_help(
            name="Chunk Overlap Segments",
            behavior="Repeats a small number of transcript segments from previous chunk into the next chunk.",
            impact="Higher overlap can catch ideas crossing chunk borders, but increases scan load and runtime.",
        )
        chunk_overlap_segments = IntPrompt.ask(
            "Chunk overlap segments (0-20)",
            default=int(runtime_defaults.get("chunk_overlap_segments", config_template["runtime"]["chunk_overlap_segments"])),
            show_default=True,
        )
        self._show_setting_help(
            name="Enable Bridge Chunks",
            behavior="Adds extra transition chunks from neighboring chunk edges.",
            impact="Enabled improves cross-boundary recall but can significantly increase total chunk count and scanning time.",
        )
        enable_bridge_chunks = Confirm.ask(
            "Enable bridge chunks?",
            default=bool(runtime_defaults.get("enable_bridge_chunks", config_template["runtime"].get("enable_bridge_chunks", True))),
        )
        bridge_chunk_edge_segments = int(
            runtime_defaults.get("bridge_chunk_edge_segments", config_template["runtime"]["bridge_chunk_edge_segments"])
        )
        if enable_bridge_chunks:
            self._show_setting_help(
                name="Bridge Edge Segments",
                behavior="How many edge segments to take from left and right chunks to build each bridge chunk.",
                impact="Higher values provide more boundary context but add extra AI scans and processing time.",
            )
            bridge_chunk_edge_segments = IntPrompt.ask(
                "Bridge edge segments (1-20)",
                default=bridge_chunk_edge_segments,
                show_default=True,
            )
        self._show_setting_help(
            name="Temperature",
            behavior="Controls generation randomness in model responses.",
            impact="Lower values improve deterministic JSON format stability; higher values can increase variation but also parse failures.",
        )
        temperature = FloatPrompt.ask("Model temperature (0.0 - 1.0)", default=0.2, show_default=True)
        self._show_setting_help(
            name="Developer Logging",
            behavior="Switch between concise production logs and verbose debug logs.",
            impact="Dev mode helps diagnostics but produces much noisier logs.",
        )
        dev_mode = Confirm.ask("Enable developer mode logging?", default=False)

        self._show_setting_help(
            name="Temp Cleanup Policy",
            behavior="Defines how archived processed source files in temp/ are retained.",
            impact="Never keeps everything; size/age modes auto-delete older files to control disk usage.",
        )
        cleanup_mode_choice = Prompt.ask(
            "Processed video retention policy",
            choices=["never", "max_size_gb", "max_age_days"],
            default="never",
        )
        cleanup_size_gb = 20
        cleanup_age_days = 30
        if cleanup_mode_choice == "max_size_gb":
            self._show_setting_help(
                name="Max Temp Size (GB)",
                behavior="Threshold for automatic deletion of oldest temp files.",
                impact="Lower values free disk earlier; very low values may remove files you expected to keep.",
            )
            cleanup_size_gb = IntPrompt.ask("Delete oldest files when temp exceeds (GB)", default=20)
        elif cleanup_mode_choice == "max_age_days":
            self._show_setting_help(
                name="Max Temp Age (Days)",
                behavior="Maximum retention age for processed temp files.",
                impact="Lower values keep temp storage cleaner but reduce how long old processed files remain available.",
            )
            cleanup_age_days = IntPrompt.ask("Delete temp files older than (days)", default=30)

        self.console.print(
            Panel(
                "Channel fetching is optional and configured later in Dashboard > Settings > Channels.",
                title="Optional Later Setup",
                border_style="cyan",
            )
        )

        system_prompt = str(
            existing_config.get("clipping", {}).get("system_prompt", DEFAULT_SYSTEM_PROMPT)
        ).strip() or DEFAULT_SYSTEM_PROMPT
        self._show_step("System Prompt")
        self._show_setting_help(
            name="System Prompt",
            behavior="Low-level instruction contract for model output format and clipping behavior.",
            impact="Bad edits can reduce relevance or break strict JSON output. Keep format constraints explicit.",
        )
        self.console.print("[bold cyan]Opening editor for system prompt...[/bold cyan]")
        system_prompt = self._read_system_prompt(current_prompt=system_prompt)

        token_plan = build_token_plan(
            recommendation=selected_recommendation,
            hardware_profile=hardware_profile,
            intensity=intensity,
        )

        config = build_default_config()
        config["app"]["dev_mode"] = bool(dev_mode)
        config["app"]["log_level"] = "DEBUG" if dev_mode else "INFO"
        config["ollama"]["url"] = ollama_url
        config["ollama"]["model"] = selected_model
        config["ollama"]["context_window"] = token_plan["total_context_tokens"]
        config["ollama"]["max_output_tokens"] = token_plan["max_output_tokens"]
        config["ollama"]["temperature"] = max(0.0, min(1.0, float(temperature)))
        config["transcription"]["model"] = whisper_model
        config["paths"]["output_dir"] = output_dir
        config["clipping"]["user_query"] = user_query
        config["clipping"]["system_prompt"] = system_prompt
        config["clipping"]["merge_distance_seconds"] = max(0, int(merge_distance_seconds))
        config["clipping"]["ai_loops"] = max(1, int(ai_loops))
        config["runtime"]["setup_intensity"] = intensity
        config["runtime"]["total_context_tokens"] = token_plan["total_context_tokens"]
        config["runtime"]["max_chunk_tokens"] = token_plan["max_chunk_tokens"]
        config["runtime"]["chunk_overlap_segments"] = max(0, int(chunk_overlap_segments))
        config["runtime"]["enable_bridge_chunks"] = bool(enable_bridge_chunks)
        config["runtime"]["bridge_chunk_edge_segments"] = max(1, int(bridge_chunk_edge_segments))
        config["runtime"]["enable_temp_run_save"] = True
        config["clipping"]["rerun_temp_files"] = bool(config["runtime"]["enable_temp_run_save"])
        prompt_cap = prompt_aware_chunk_cap(
            configured_chunk_tokens=int(config["runtime"]["max_chunk_tokens"]),
            total_context_tokens=int(config["runtime"]["total_context_tokens"]),
            max_output_tokens=int(config["ollama"]["max_output_tokens"]),
            user_query=user_query,
            system_prompt=system_prompt,
        )
        config["runtime"]["max_chunk_tokens"] = int(prompt_cap["effective_chunk_tokens"])
        config["maintenance"]["temp_cleanup"]["mode"] = cleanup_mode_choice
        config["maintenance"]["temp_cleanup"]["max_size_gb"] = max(1, int(cleanup_size_gb))
        config["maintenance"]["temp_cleanup"]["max_age_days"] = max(1, int(cleanup_age_days))

        validation_errors = validate_config(config)
        if validation_errors:
            raise ValueError(f"Generated config is invalid: {validation_errors}")

        model_profile = {
            "selected_model": selected_model,
            "thinking_model": selected_recommendation.thinking,
            "selection_source": selected_recommendation.source,
            "selection_reason": selected_recommendation.reason,
            "estimated_model_size_gb": selected_recommendation.estimated_size_gb,
            "token_plan": token_plan,
            "intensity": intensity,
            "local_models_seen": [model.get("name", "") for model in local_models],
        }
        hardware_profile["selected_intensity"] = intensity

        self._show_step("Finalizing Setup")
        summary = Table(title="Configuration Summary")
        summary.add_column("Setting", style="cyan")
        summary.add_column("Value", style="white")
        summary.add_row("Model", selected_model)
        summary.add_row("Whisper", whisper_model)
        summary.add_row("Output Directory", output_dir)
        summary.add_row("Intensity", intensity)
        summary.add_row("Chunk Overlap", str(config["runtime"]["chunk_overlap_segments"]))
        summary.add_row("Bridge Chunks", "Enabled" if config["runtime"]["enable_bridge_chunks"] else "Disabled")
        if config["runtime"]["enable_bridge_chunks"]:
            summary.add_row("Bridge Edge Segments", str(config["runtime"]["bridge_chunk_edge_segments"]))
        summary.add_row("Dev Logging", "Enabled" if dev_mode else "Disabled")
        summary.add_row("Temp Cleanup", cleanup_mode_choice)
        self.console.print(summary)

        write_config_bundle(
            base_dir=self.base_dir,
            config=config,
            hardware_profile=hardware_profile,
            model_profile=model_profile,
        )

        self.console.print(success_panel("Setup complete. Configuration files were generated in config/."))
        return config
