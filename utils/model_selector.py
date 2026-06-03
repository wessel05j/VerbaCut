from __future__ import annotations

import logging
import math
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from utils.validators import normalize_ollama_url


THINKING_HINTS = ("gpt-oss", "r1", "qwq", "reason", "o1", "o3")
MIN_THINKING_B = 8.0
POLICY_MODEL = "gpt-oss:20b"

THINKING_CATALOG: List[Dict[str, Any]] = [
    {"name": "gpt-oss:20b", "size_gb": 13.0, "thinking": True, "context": 16384, "think_low": True},
    {"name": "qwq:32b", "size_gb": 20.0, "thinking": True, "context": 16384, "think_low": False},
    {"name": "deepseek-r1:14b", "size_gb": 9.0, "thinking": True, "context": 12288, "think_low": False},
    {"name": "deepseek-r1:8b", "size_gb": 5.0, "thinking": True, "context": 8192, "think_low": False},
]


@dataclass
class ModelRecommendation:
    model_name: str
    thinking: bool
    source: str
    estimated_size_gb: float
    reason: str
    context_window: int
    max_output_tokens: int


def is_thinking_model(model_name: str) -> bool:
    lowered = model_name.lower()
    return any(hint in lowered for hint in THINKING_HINTS)


def model_billions(model_name: str) -> Optional[float]:
    match = re.search(r"(\d+(?:\.\d+)?)b", model_name.lower())
    if not match:
        return None
    try:
        return float(match.group(1))
    except Exception:
        return None


def is_required_thinking_model(model_name: str) -> bool:
    if not is_thinking_model(model_name):
        return False
    billions = model_billions(model_name)
    if billions is None:
        return False
    return billions >= MIN_THINKING_B


def supports_think_low(model_name: str) -> bool:
    return str(model_name or "").lower().startswith("gpt-oss")


def _guess_size_from_name(model_name: str) -> Optional[float]:
    billions = model_billions(model_name)
    if billions is None:
        return None
    return max(1.5, round(billions * 0.62, 1))


def _model_context_hint(model_name: str, fallback: int) -> int:
    for model in THINKING_CATALOG:
        if model["name"].lower() == model_name.lower():
            return int(model["context"])
    return fallback


def ollama_api_reachable(ollama_url: str, timeout: float = 3.0) -> bool:
    url = normalize_ollama_url(ollama_url)
    endpoints = (f"{url}/api/version", f"{url}/api/tags")
    for endpoint in endpoints:
        try:
            response = requests.get(endpoint, timeout=timeout)
        except requests.RequestException:
            continue
        if response.ok:
            return True
    return False


def _dedupe_paths(paths: List[Path]) -> List[Path]:
    seen: set[str] = set()
    deduped: List[Path] = []
    for path in paths:
        key = str(path).lower() if os.name == "nt" else str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def _windows_ollama_dir_candidates() -> List[Path]:
    candidates: List[Path] = []
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        candidates.append(Path(local_app_data) / "Programs" / "Ollama")

    for env_name in ("ProgramFiles", "ProgramFiles(x86)"):
        root = os.environ.get(env_name)
        if root:
            candidates.append(Path(root) / "Ollama")

    return _dedupe_paths(candidates)


def _find_ollama_cli() -> Optional[str]:
    discovered = shutil.which("ollama")
    if discovered:
        return discovered

    candidates: List[Path] = []
    if os.name == "nt":
        candidates.extend(path / "ollama.exe" for path in _windows_ollama_dir_candidates())
    else:
        candidates.extend(
            Path(path)
            for path in (
                "/opt/homebrew/bin/ollama",
                "/usr/local/bin/ollama",
                "/usr/bin/ollama",
            )
        )

    for candidate in _dedupe_paths(candidates):
        if candidate.exists():
            return str(candidate)
    return None


def _find_windows_ollama_app() -> Optional[str]:
    if os.name != "nt":
        return None

    for install_dir in _windows_ollama_dir_candidates():
        for filename in ("ollama app.exe", "Ollama app.exe", "Ollama.exe"):
            candidate = install_dir / filename
            if candidate.exists():
                return str(candidate)
    return None


def _ollama_start_commands() -> List[tuple[str, List[str]]]:
    commands: List[tuple[str, List[str]]] = []

    cli = _find_ollama_cli()
    if cli:
        commands.append(("ollama serve", [cli, "serve"]))

    if os.name == "nt":
        app = _find_windows_ollama_app()
        if app:
            commands.append(("Ollama Windows app", [app]))
    elif os.name == "posix" and hasattr(os, "uname") and os.uname().sysname.lower() == "darwin":
        if shutil.which("open"):
            commands.append(("Ollama macOS app", ["open", "-a", "Ollama"]))

    return commands


def _start_ollama_process(command: List[str], logger: logging.Logger) -> bool:
    startup_kwargs: Dict[str, Any] = {
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if os.name == "nt":
        startup_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
    else:
        startup_kwargs["start_new_session"] = True

    try:
        subprocess.Popen(command, **startup_kwargs)
    except Exception as exc:
        logger.warning("Unable to start Ollama command `%s`: %s", " ".join(command), exc)
        return False
    return True


def ensure_ollama_running(ollama_url: str, logger: logging.Logger, timeout_seconds: int = 45) -> bool:
    if ollama_api_reachable(ollama_url):
        return True

    commands = _ollama_start_commands()
    if not commands:
        logger.error("Ollama is not reachable and no Ollama executable could be found.")
        return False

    wait_per_attempt = max(5, int(timeout_seconds / max(1, len(commands))))
    for label, command in commands:
        logger.info("Ollama service not detected, trying to start %s.", label)
        if not _start_ollama_process(command, logger):
            continue

        deadline = time.time() + wait_per_attempt
        while time.time() < deadline:
            if ollama_api_reachable(ollama_url):
                logger.info("Ollama is ready at %s.", normalize_ollama_url(ollama_url))
                return True
            time.sleep(1.0)

    return ollama_api_reachable(ollama_url)


def fetch_local_models(ollama_url: str, timeout: float = 10.0) -> List[Dict[str, Any]]:
    url = normalize_ollama_url(ollama_url)
    try:
        response = requests.get(f"{url}/api/tags", timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return []

    models = payload.get("models", [])
    if not isinstance(models, list):
        return []
    parsed_models: List[Dict[str, Any]] = []
    for item in models:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("model") or "").strip()
        if not name:
            continue
        size_bytes = item.get("size")
        try:
            size_gb = round(float(size_bytes) / (1024**3), 2) if size_bytes else None
        except Exception:
            size_gb = None
        parsed_models.append(
            {
                "name": name,
                "size_gb": size_gb,
                "thinking": is_thinking_model(name),
                "raw": item,
            }
        )
    return parsed_models


def model_exists_locally(model_name: str, local_models: List[Dict[str, Any]]) -> bool:
    lowered = model_name.lower()
    for model in local_models:
        if str(model.get("name", "")).lower() == lowered:
            return True
    return False


def pull_model(model_name: str, logger: logging.Logger) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            ["ollama", "pull", model_name],
            capture_output=True,
            check=False,
        )
    except Exception as exc:
        logger.error("Failed to execute `ollama pull %s`: %s", model_name, exc)
        return False, str(exc)

    stdout = (result.stdout or b"").decode("utf-8", errors="replace").strip()
    stderr = (result.stderr or b"").decode("utf-8", errors="replace").strip()
    output = "\n".join(part for part in (stdout, stderr) if part)
    if result.returncode == 0:
        return True, output
    return False, output or f"ollama pull returned code {result.returncode}"


def estimate_budget_gb(hardware_profile: Dict[str, Any], intensity: str) -> float:
    intensity_key = intensity.lower().strip()
    gpu_vram_gb = float(hardware_profile.get("gpu_vram_gb", 0.0) or 0.0)
    ram_gb = float(hardware_profile.get("ram_gb", 0.0) or 0.0)

    if gpu_vram_gb > 0:
        multiplier = {"light": 0.45, "balanced": 0.64, "maximum": 0.82}.get(intensity_key, 0.64)
        return max(2.0, round(gpu_vram_gb * multiplier, 1))

    multiplier = {"light": 0.18, "balanced": 0.28, "maximum": 0.40}.get(intensity_key, 0.28)
    return max(2.0, round(ram_gb * multiplier, 1))


def recommend_model(
    hardware_profile: Dict[str, Any],
    intensity: str,
    local_models: List[Dict[str, Any]],
) -> ModelRecommendation:
    budget_gb = estimate_budget_gb(hardware_profile, intensity)
    base_context = int(hardware_profile.get("recommended_context_window", 8192) or 8192)
    base_context = max(4096, min(32768, base_context))
    policy_meta = next((item for item in THINKING_CATALOG if item["name"].lower() == POLICY_MODEL.lower()), None)
    policy_size = float(policy_meta["size_gb"]) if policy_meta else 13.0
    policy_ctx = int(policy_meta["context"]) if policy_meta else 16384

    installed_policy = next(
        (item for item in local_models if str(item.get("name", "")).lower() == POLICY_MODEL.lower()),
        None,
    )
    policy_installed = installed_policy is not None
    if policy_installed:
        installed_size = installed_policy.get("size_gb")
        if isinstance(installed_size, (int, float)):
            policy_size = float(installed_size)

    pressure = policy_size / max(1.0, budget_gb)
    if pressure <= 1.0:
        reason = (
            f"Policy default `{POLICY_MODEL}` is recommended and fits estimated budget "
            f"({budget_gb:.1f} GB)."
        )
    else:
        reason = (
            f"Policy default `{POLICY_MODEL}` is recommended even though estimated pressure is high "
            f"({policy_size:.1f} GB model vs ~{budget_gb:.1f} GB budget). "
            "You can still proceed, but expect slower processing."
        )

    max_output = 1000
    return ModelRecommendation(
        model_name=POLICY_MODEL,
        thinking=True,
        source="policy-default-local" if policy_installed else "policy-default-catalog",
        estimated_size_gb=policy_size,
        reason=reason,
        context_window=min(policy_ctx, base_context),
        max_output_tokens=max_output,
    )


def build_token_plan(
    recommendation: ModelRecommendation,
    hardware_profile: Dict[str, Any],
    intensity: str,
) -> Dict[str, int]:
    max_tokens = int(hardware_profile.get("max_tokens_estimate", 8192) or 8192)
    ctx_limit = max(4096, min(max_tokens, recommendation.context_window))

    intensity_key = intensity.lower().strip()
    safety = {"light": 0.35, "balanced": 0.25, "maximum": 0.2}.get(intensity_key, 0.25)
    reserved = int(max(700, recommendation.max_output_tokens + (ctx_limit * safety)))
    max_chunk_tokens = max(512, ctx_limit - reserved)

    return {
        "total_context_tokens": ctx_limit,
        "max_output_tokens": recommendation.max_output_tokens,
        "max_chunk_tokens": max_chunk_tokens,
    }


def build_runtime_token_plan(
    model_name: str,
    context_window: int,
    max_output_tokens: int,
    hardware_profile: Dict[str, Any],
    intensity: str,
) -> Dict[str, int]:
    guessed_size = _guess_size_from_name(model_name) or 10.0
    recommendation = ModelRecommendation(
        model_name=model_name,
        thinking=is_thinking_model(model_name),
        source="runtime-update",
        estimated_size_gb=float(guessed_size),
        reason="Runtime token recalculation.",
        context_window=int(max(1024, context_window)),
        max_output_tokens=int(max(128, max_output_tokens)),
    )
    return build_token_plan(
        recommendation=recommendation,
        hardware_profile=hardware_profile,
        intensity=intensity,
    )


def _estimate_tokens_rough(text: str) -> int:
    payload = str(text or "")
    return max(1, math.floor(len(payload) / 3.5))


def prompt_aware_chunk_cap(
    configured_chunk_tokens: int,
    total_context_tokens: int,
    max_output_tokens: int,
    user_query: str,
    system_prompt: str,
    reserved_tokens: int = 700,
) -> Dict[str, int]:
    prompt_tokens = (
        _estimate_tokens_rough(user_query)
        + _estimate_tokens_rough(system_prompt)
        + int(max(0, reserved_tokens))
    )
    available = int(total_context_tokens) - int(max_output_tokens) - int(prompt_tokens)
    if available >= 900:
        effective = min(int(configured_chunk_tokens), int(available))
        effective = max(900, int(effective))
    else:
        effective = max(256, min(int(configured_chunk_tokens), int(max(256, available))))
    return {
        "effective_chunk_tokens": int(effective),
        "prompt_tokens": int(prompt_tokens),
        "available_context_tokens": int(available),
    }
