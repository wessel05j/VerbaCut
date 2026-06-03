from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


APP_NAME = "VerbaCut"
APP_VERSION = "2.0.0"
APP_GITHUB = "https://github.com/wessel05j/VerbaCut"

CONFIG_DIR = "config"
CONFIG_FILE = "config.json"
HARDWARE_PROFILE_FILE = "hardware_profile.json"
MODEL_PROFILE_FILE = "model_profile.json"
THINKING_HINTS = ("gpt-oss", "r1", "qwq", "reason", "o1", "o3")
MIN_THINKING_B = 8.0

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert transcript clip selector for the users request.\n"
    "Input:\n"
    "- Transcript chunk lines formatted as: segment_id | start | end | text\n"
    "- start/end values are in seconds.\n\n"
    "OUTPUT ONLY (STRICT):\n"
    "- ONLY JSON: [[start1, end1, score1], [start2, end2, score2], ...]\n"
    "- The third element is how well the clip matches the user query "
    "(1-10, 1=good, 5=very good and 10=excellent).\n"
    "- No explanations, no extra keys, no comments.\n\n"
    "Rules:\n"
    "- Keep full timestamp precision; do not round.\n"
    "- Each clip must be a complete thought: clear beginning, middle, and end.\n"
    "- Prefer natural pauses and paragraph boundaries for cut points.\n"
    "- Prefer context-rich clips that clearly match the user query.\n"
    "- If no segments match the query, return an empty list: [].\n"
    "- Never return a blank response, {}, null, or any JSON object.\n"
)

DEFAULT_USER_QUERY = (
    "Find the most engaging and context-complete moments that work as viral short clips."
)


def ensure_runtime_layout(base_dir: Path) -> Dict[str, Path]:
    """Create required runtime folders and keep-empty markers."""
    paths = {
        "input": base_dir / "input",
        "output": base_dir / "output",
        "temp": base_dir / "temp",
        "system": base_dir / "system",
        "config": base_dir / "config",
        "logs": base_dir / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    (paths["config"] / ".gitkeep").touch(exist_ok=True)
    (paths["logs"] / ".gitkeep").touch(exist_ok=True)
    return paths


def normalize_ollama_url(url: str) -> str:
    normalized = (url or "http://localhost:11434").strip()
    if not normalized:
        normalized = "http://localhost:11434"
    if not normalized.startswith(("http://", "https://")):
        normalized = f"http://{normalized}"
    return normalized.rstrip("/")


def load_json_file(path: Path) -> Any:
    # Accept optional UTF-8 BOM to avoid startup failures from external editors/tools.
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def save_json_file(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_fd, temp_path = tempfile.mkstemp(
        prefix=f"{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(temp_fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=True)
        Path(temp_path).replace(path)
    finally:
        try:
            Path(temp_path).unlink(missing_ok=True)
        except Exception:
            pass


def build_default_config() -> Dict[str, Any]:
    return {
        "app": {
            "name": APP_NAME,
            "version": APP_VERSION,
            "github": APP_GITHUB,
            "dev_mode": False,
            "log_level": "INFO",
        },
        "paths": {
            "input_dir": "input",
            "output_dir": "output",
            "temp_dir": "temp",
            "system_dir": "system",
            "logs_dir": "logs",
        },
        "ollama": {
            "url": "http://localhost:11434",
            "model": "gpt-oss:20b",
            "context_window": 8192,
            "max_output_tokens": 900,
            "temperature": 0.2,
        },
        "transcription": {
            "model": "base",
        },
        "clipping": {
            "user_query": DEFAULT_USER_QUERY,
            "system_prompt": DEFAULT_SYSTEM_PROMPT,
            "merge_distance_seconds": 20,
            "ai_loops": 2,
            "enable_youtube_downloads": True,
            "youtube_links": [],
            "channels": [],
            "channels_hours_limit": 24,
            "rerun_temp_files": True,
            "clip_progress_interval": 5,
        },
        "runtime": {
            "total_context_tokens": 8192,
            "max_chunk_tokens": 6000,
            "setup_intensity": "balanced",
            "chunk_overlap_segments": 3,
            "enable_bridge_chunks": True,
            "bridge_chunk_edge_segments": 4,
            "enable_temp_run_save": True,
        },
        "maintenance": {
            "temp_cleanup": {
                "mode": "never",
                "max_size_gb": 20,
                "max_age_days": 30,
            }
        },
    }


def migrate_legacy_settings(base_dir: Path) -> Optional[Dict[str, Any]]:
    """Map legacy `system/settings.json` into new config schema when available."""
    legacy_settings_path = base_dir / "system" / "settings.json"
    if not legacy_settings_path.exists():
        return None

    try:
        legacy = load_json_file(legacy_settings_path)
    except Exception:
        return None

    config = build_default_config()
    config["ollama"]["url"] = normalize_ollama_url(legacy.get("ollama_url", "http://localhost:11434"))
    config["ollama"]["model"] = str(legacy.get("ai_model") or config["ollama"]["model"])
    config["ollama"]["max_output_tokens"] = int(legacy.get("max_ai_tokens", config["ollama"]["max_output_tokens"]))
    config["ollama"]["temperature"] = float(legacy.get("temperature", config["ollama"]["temperature"]))
    config["transcription"]["model"] = str(
        legacy.get("transcribing_model", config["transcription"]["model"])
    ).lower()
    config["clipping"]["user_query"] = str(
        legacy.get("user_query", config["clipping"]["user_query"])
    ).strip() or DEFAULT_USER_QUERY
    config["clipping"]["system_prompt"] = str(
        legacy.get("system_query", config["clipping"]["system_prompt"])
    ).strip() or DEFAULT_SYSTEM_PROMPT
    config["clipping"]["merge_distance_seconds"] = int(
        legacy.get("merge_distance", config["clipping"]["merge_distance_seconds"])
    )
    config["clipping"]["ai_loops"] = int(legacy.get("ai_loops", config["clipping"]["ai_loops"]))
    config["clipping"]["youtube_links"] = list(legacy.get("youtube_list", []))
    config["clipping"]["channels"] = list(legacy.get("channels", []))
    config["clipping"]["channels_hours_limit"] = int(
        legacy.get("channels_hours_limit", config["clipping"]["channels_hours_limit"])
    )
    config["clipping"]["enable_youtube_downloads"] = bool(
        legacy.get("enable_youtube_downloads", config["clipping"]["enable_youtube_downloads"])
    )
    config["clipping"]["rerun_temp_files"] = bool(
        legacy.get("rerun_temp_files", config["clipping"]["rerun_temp_files"])
    )
    config["runtime"]["total_context_tokens"] = int(
        legacy.get("total_tokens", config["runtime"]["total_context_tokens"])
    )
    config["runtime"]["max_chunk_tokens"] = int(
        legacy.get("max_chunking_tokens", config["runtime"]["max_chunk_tokens"])
    )
    config["runtime"]["enable_temp_run_save"] = bool(
        legacy.get("rerun_temp_files", config["runtime"]["enable_temp_run_save"])
    )
    return config


def _validate_range(errors: List[str], name: str, value: Any, minimum: float, maximum: float) -> None:
    try:
        numeric = float(value)
    except Exception:
        errors.append(f"{name} must be numeric")
        return
    if numeric < minimum or numeric > maximum:
        errors.append(f"{name} must be between {minimum} and {maximum}")


def _model_billions(model_name: str) -> Optional[float]:
    import re

    match = re.search(r"(\d+(?:\.\d+)?)b", model_name.lower())
    if not match:
        return None
    try:
        return float(match.group(1))
    except Exception:
        return None


def _is_required_thinking_model(model_name: str) -> bool:
    lowered = model_name.lower()
    if not any(hint in lowered for hint in THINKING_HINTS):
        return False
    billions = _model_billions(model_name)
    if billions is None:
        return False
    return billions >= MIN_THINKING_B


def validate_config(config: Dict[str, Any]) -> List[str]:
    errors: List[str] = []

    for key in ("app", "paths", "ollama", "transcription", "clipping", "runtime", "maintenance"):
        if key not in config or not isinstance(config[key], dict):
            errors.append(f"Missing or invalid object: {key}")

    if errors:
        return errors

    model_name = str(config["ollama"].get("model", "")).strip()
    if not model_name:
        errors.append("ollama.model cannot be empty")
    elif not _is_required_thinking_model(model_name):
        errors.append("ollama.model must be a thinking model with at least 8b parameters")

    for path_key in ("input_dir", "output_dir", "temp_dir", "system_dir", "logs_dir"):
        path_value = str(config["paths"].get(path_key, "")).strip()
        if not path_value:
            errors.append(f"paths.{path_key} cannot be empty")

    config["ollama"]["url"] = normalize_ollama_url(str(config["ollama"].get("url", "")))
    _validate_range(errors, "ollama.temperature", config["ollama"].get("temperature"), 0.0, 1.0)
    _validate_range(errors, "ollama.context_window", config["ollama"].get("context_window"), 1024, 131072)
    _validate_range(
        errors,
        "ollama.max_output_tokens",
        config["ollama"].get("max_output_tokens"),
        128,
        16384,
    )

    whisper_model = str(config["transcription"].get("model", "")).lower().strip()
    if whisper_model not in {"tiny", "base", "small", "medium", "large"}:
        errors.append("transcription.model must be one of tiny/base/small/medium/large")

    user_query = str(config["clipping"].get("user_query", "")).strip()
    if not user_query:
        errors.append("clipping.user_query cannot be empty")
    system_prompt = str(config["clipping"].get("system_prompt", "")).strip()
    if not system_prompt:
        errors.append("clipping.system_prompt cannot be empty")

    _validate_range(
        errors,
        "clipping.merge_distance_seconds",
        config["clipping"].get("merge_distance_seconds"),
        0,
        600,
    )
    _validate_range(errors, "clipping.ai_loops", config["clipping"].get("ai_loops"), 1, 20)
    _validate_range(
        errors,
        "clipping.channels_hours_limit",
        config["clipping"].get("channels_hours_limit"),
        1,
        336,
    )

    if not isinstance(config["clipping"].get("youtube_links", []), list):
        errors.append("clipping.youtube_links must be a list")
    if not isinstance(config["clipping"].get("channels", []), list):
        errors.append("clipping.channels must be a list")
    if "enable_youtube_downloads" not in config["clipping"]:
        config["clipping"]["enable_youtube_downloads"] = True
    if not isinstance(config["clipping"].get("enable_youtube_downloads", True), bool):
        errors.append("clipping.enable_youtube_downloads must be true/false")

    log_level = str(config["app"].get("log_level", "")).upper()
    if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
        errors.append("app.log_level must be DEBUG/INFO/WARNING/ERROR")

    intensity = str(config["runtime"].get("setup_intensity", "")).lower()
    if intensity not in {"light", "balanced", "maximum"}:
        errors.append("runtime.setup_intensity must be light/balanced/maximum")

    _validate_range(
        errors,
        "runtime.total_context_tokens",
        config["runtime"].get("total_context_tokens"),
        1024,
        131072,
    )
    _validate_range(
        errors,
        "runtime.max_chunk_tokens",
        config["runtime"].get("max_chunk_tokens"),
        256,
        120000,
    )
    _validate_range(
        errors,
        "runtime.chunk_overlap_segments",
        config["runtime"].get("chunk_overlap_segments", 3),
        0,
        20,
    )
    if not isinstance(config["runtime"].get("enable_bridge_chunks", True), bool):
        errors.append("runtime.enable_bridge_chunks must be true/false")
    if "enable_temp_run_save" not in config["runtime"]:
        config["runtime"]["enable_temp_run_save"] = bool(
            config["clipping"].get("rerun_temp_files", True)
        )
    if not isinstance(config["runtime"].get("enable_temp_run_save", True), bool):
        errors.append("runtime.enable_temp_run_save must be true/false")
    config["clipping"]["rerun_temp_files"] = bool(config["runtime"].get("enable_temp_run_save", True))
    _validate_range(
        errors,
        "runtime.bridge_chunk_edge_segments",
        config["runtime"].get("bridge_chunk_edge_segments", 4),
        1,
        20,
    )

    temp_cleanup = config["maintenance"].get("temp_cleanup", {})
    if not isinstance(temp_cleanup, dict):
        errors.append("maintenance.temp_cleanup must be an object")
    else:
        mode = str(temp_cleanup.get("mode", "")).lower().strip()
        if mode not in {"never", "max_size_gb", "max_age_days"}:
            errors.append("maintenance.temp_cleanup.mode must be never/max_size_gb/max_age_days")
        _validate_range(
            errors,
            "maintenance.temp_cleanup.max_size_gb",
            temp_cleanup.get("max_size_gb", 20),
            1,
            100000,
        )
        _validate_range(
            errors,
            "maintenance.temp_cleanup.max_age_days",
            temp_cleanup.get("max_age_days", 30),
            1,
            3650,
        )

    return errors


def config_paths(base_dir: Path) -> Dict[str, Path]:
    config_dir = base_dir / CONFIG_DIR
    return {
        "config_dir": config_dir,
        "config_file": config_dir / CONFIG_FILE,
        "hardware_file": config_dir / HARDWARE_PROFILE_FILE,
        "model_file": config_dir / MODEL_PROFILE_FILE,
    }


def load_validated_config(base_dir: Path) -> Tuple[bool, Optional[Dict[str, Any]], List[str]]:
    paths = config_paths(base_dir)
    config_file = paths["config_file"]

    if not config_file.exists():
        migrated = migrate_legacy_settings(base_dir)
        if migrated:
            errors = validate_config(migrated)
            if not errors:
                save_json_file(config_file, migrated)
                return True, migrated, []
        return False, None, ["config/config.json not found"]

    try:
        config = load_json_file(config_file)
    except Exception as exc:
        return False, None, [f"Unable to read config/config.json: {exc}"]

    if not isinstance(config, dict):
        return False, None, ["config/config.json must be a JSON object"]

    before_normalization = json.dumps(config, sort_keys=True, ensure_ascii=True)
    errors = validate_config(config)
    if errors:
        return False, config, errors
    after_normalization = json.dumps(config, sort_keys=True, ensure_ascii=True)
    if after_normalization != before_normalization:
        save_json_file(config_file, config)
    return True, config, []


def write_config_bundle(
    base_dir: Path,
    config: Dict[str, Any],
    hardware_profile: Dict[str, Any],
    model_profile: Dict[str, Any],
) -> None:
    paths = config_paths(base_dir)
    save_json_file(paths["config_file"], config)
    save_json_file(paths["hardware_file"], hardware_profile)
    save_json_file(paths["model_file"], model_profile)


def load_optional_profile(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = load_json_file(path)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}
