#!/usr/bin/env python
"""
Environment bootstrapper for AI Auto Clipper.

What it does:
1) Installs/updates Python dependencies from requirements.txt
2) Installs torch in CPU or CUDA mode (auto-detected by default)
3) Saves setup state so repeated launches can skip reinstall work
4) Warns if ffmpeg or ollama CLI is missing
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


BASE_DIR = Path(__file__).resolve().parent
SYSTEM_DIR = BASE_DIR / "system"
STATE_FILE = SYSTEM_DIR / "setup_state.json"
REQUIREMENTS_FILE = BASE_DIR / "requirements.txt"
CUDA_INDEX_URLS = (
    "https://download.pytorch.org/whl/cu124",
    "https://download.pytorch.org/whl/cu121",
    "https://download.pytorch.org/whl/cu118",
)

WINDOWS_TOOL_PLAN: Dict[str, Dict[str, Any]] = {
    "ffmpeg": {"label": "FFmpeg", "id": "Gyan.FFmpeg", "estimate_mb": 180, "required": True},
    "ollama": {"label": "Ollama", "id": "Ollama.Ollama", "estimate_mb": 1200, "required": True},
    "node": {"label": "Node.js LTS", "id": "OpenJS.NodeJS.LTS", "estimate_mb": 80, "required": False},
}

MAC_TOOL_PLAN: Dict[str, Dict[str, Any]] = {
    "ffmpeg": {"label": "FFmpeg", "brew": "ffmpeg", "estimate_mb": 180, "required": True},
    "ollama": {"label": "Ollama", "brew": "--cask ollama", "estimate_mb": 1200, "required": True},
    "node": {"label": "Node.js LTS", "brew": "node", "estimate_mb": 80, "required": False},
}

LINUX_TOOL_PLAN: Dict[str, Dict[str, Any]] = {
    "ffmpeg": {"label": "FFmpeg", "estimate_mb": 180, "required": True},
    "ollama": {"label": "Ollama", "estimate_mb": 1200, "required": True},
    "node": {"label": "Node.js LTS", "estimate_mb": 80, "required": False},
}


def run_cmd(args: list[str], capture_output: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        args,
        check=True,
        text=True,
        capture_output=capture_output,
    )


def pip_install(args: list[str]) -> None:
    run_cmd([sys.executable, "-m", "pip", *args])


def read_setup_state() -> Dict[str, Any]:
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def write_setup_state(state: Dict[str, Any]) -> None:
    SYSTEM_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def requirements_hash() -> str:
    payload = REQUIREMENTS_FILE.read_bytes()
    return hashlib.sha256(payload).hexdigest()


def running_in_venv() -> bool:
    return sys.prefix != getattr(sys, "base_prefix", sys.prefix)


def detect_nvidia_gpu() -> bool:
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        result = run_cmd(["nvidia-smi", "-L"], capture_output=True)
    except Exception:
        return False
    return bool(result.stdout.strip())


def probe_torch() -> Dict[str, Any]:
    if importlib.util.find_spec("torch") is None:
        return {"installed": False}
    try:
        import torch  # type: ignore

        return {
            "installed": True,
            "version": getattr(torch, "__version__", "unknown"),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
        }
    except Exception as exc:  # pragma: no cover - defensive path for broken envs
        return {"installed": False, "error": str(exc)}


def resolve_torch_target(requested: str, has_nvidia_gpu: bool) -> str:
    if requested == "auto":
        return "cuda" if has_nvidia_gpu else "cpu"
    return requested


def torch_satisfies_target(torch_info: Dict[str, Any], target: str) -> bool:
    if target == "skip":
        return True
    if not torch_info.get("installed"):
        return False
    if target == "cuda":
        return bool(torch_info.get("cuda_available"))
    # CPU target accepts either CPU-only torch or CUDA-capable torch.
    return True


def install_torch(target: str) -> Tuple[str, Dict[str, Any]]:
    if target == "skip":
        return "skip", probe_torch()

    if target == "cpu":
        print("Installing torch (CPU build)...")
        pip_install(["install", "--upgrade", "torch"])
        return "cpu", probe_torch()

    if target == "cuda":
        for index_url in CUDA_INDEX_URLS:
            print(f"Installing torch CUDA build via {index_url} ...")
            try:
                pip_install(["install", "--upgrade", "torch", "--index-url", index_url])
                torch_info = probe_torch()
                if torch_info.get("cuda_available"):
                    return "cuda", torch_info
            except subprocess.CalledProcessError:
                continue

        print("CUDA torch install failed or CUDA not available. Falling back to CPU torch.")
        pip_install(["install", "--upgrade", "torch"])
        return "cpu", probe_torch()

    raise ValueError(f"Unknown torch target: {target}")


def _detect_linux_package_manager() -> Optional[str]:
    for manager in ("apt-get", "dnf", "yum", "pacman", "zypper"):
        if shutil.which(manager):
            return manager
    return None


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


def _find_ollama_executable() -> Optional[str]:
    discovered = shutil.which("ollama")
    if discovered:
        return discovered

    candidates: List[Path] = []
    if os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            candidates.append(Path(local_app_data) / "Programs" / "Ollama" / "ollama.exe")
        for env_name in ("ProgramFiles", "ProgramFiles(x86)"):
            root = os.environ.get(env_name)
            if root:
                candidates.append(Path(root) / "Ollama" / "ollama.exe")
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


def _collect_dependency_status() -> Dict[str, Any]:
    return {
        "python": shutil.which("python") is not None or shutil.which("python3") is not None,
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "ffprobe": shutil.which("ffprobe") is not None,
        "ollama": _find_ollama_executable() is not None,
        "node": shutil.which("node") is not None,
        "deno": shutil.which("deno") is not None,
        "yt_dlp_module": importlib.util.find_spec("yt_dlp") is not None,
    }


def _probe_youtube_cookie_sources() -> Tuple[bool, str]:
    cookie_candidates = (
        BASE_DIR / "resources" / "cookies.txt",
        BASE_DIR / "system" / "cookies.txt",
        BASE_DIR / "cookies.txt",
    )
    for candidate in cookie_candidates:
        if candidate.exists() and candidate.stat().st_size > 0:
            return True, f"Found cookies file at {candidate}."

    if importlib.util.find_spec("yt_dlp") is None:
        return False, "yt-dlp module is unavailable for cookie probe."

    try:
        import yt_dlp  # type: ignore
    except Exception as exc:
        return False, f"yt-dlp import failed: {exc}"

    for browser in ("firefox", "edge", "chrome", "brave", "opera", "vivaldi"):
        test_opts: Dict[str, Any] = {
            "quiet": True,
            "skip_download": True,
            "extract_flat": "in_playlist",
            "playlist_items": "1",
            "cookiesfrombrowser": (browser,),
            "ignoreerrors": True,
            "no_warnings": True,
            "ignoreconfig": True,
        }
        try:
            with yt_dlp.YoutubeDL(test_opts) as ydl:
                result = ydl.extract_info("https://www.youtube.com/@YouTube/videos", download=False)
            if result and result.get("entries"):
                return True, f"Browser cookie extraction works with {browser}."
        except Exception:
            continue
    return False, "No cookie file/browser cookie source is currently usable for YouTube."


def _build_install_commands(missing_tools: List[str]) -> Tuple[List[Tuple[str, int, Any]], List[str]]:
    commands: List[Tuple[str, int, Any]] = []
    notes: List[str] = []
    system_name = platform.system().lower()

    if system_name == "windows":
        if shutil.which("winget") is None:
            notes.append("winget is not installed. Install tools manually from vendor sites.")
            return commands, notes
        for tool in missing_tools:
            meta = WINDOWS_TOOL_PLAN.get(tool)
            if not meta:
                continue
            cmd = [
                "winget",
                "install",
                "--id",
                str(meta["id"]),
                "-e",
                "--accept-package-agreements",
                "--accept-source-agreements",
            ]
            commands.append((str(meta["label"]), int(meta["estimate_mb"]), cmd))
        return commands, notes

    if system_name == "darwin":
        if shutil.which("brew") is None:
            notes.append("Homebrew not found. Install from https://brew.sh first.")
            return commands, notes
        for tool in missing_tools:
            meta = MAC_TOOL_PLAN.get(tool)
            if not meta:
                continue
            package_arg = str(meta["brew"])
            cmd = f"brew install {package_arg}"
            commands.append((str(meta["label"]), int(meta["estimate_mb"]), cmd))
        return commands, notes

    package_manager = _detect_linux_package_manager()
    if package_manager is None:
        notes.append("No supported Linux package manager detected (apt/dnf/yum/pacman/zypper).")
        return commands, notes

    if "ffmpeg" in missing_tools:
        if package_manager == "apt-get":
            commands.append(("FFmpeg", 180, "sudo apt-get update && sudo apt-get install -y ffmpeg"))
        elif package_manager == "dnf":
            commands.append(("FFmpeg", 180, "sudo dnf install -y ffmpeg ffmpeg-libs"))
        elif package_manager == "yum":
            commands.append(("FFmpeg", 180, "sudo yum install -y ffmpeg"))
        elif package_manager == "pacman":
            commands.append(("FFmpeg", 180, "sudo pacman -S --noconfirm ffmpeg"))
        elif package_manager == "zypper":
            commands.append(("FFmpeg", 180, "sudo zypper --non-interactive install ffmpeg"))

    if "node" in missing_tools:
        if package_manager == "apt-get":
            commands.append(("Node.js LTS", 80, "sudo apt-get install -y nodejs npm"))
        elif package_manager == "dnf":
            commands.append(("Node.js LTS", 80, "sudo dnf install -y nodejs npm"))
        elif package_manager == "yum":
            commands.append(("Node.js LTS", 80, "sudo yum install -y nodejs npm"))
        elif package_manager == "pacman":
            commands.append(("Node.js LTS", 80, "sudo pacman -S --noconfirm nodejs npm"))
        elif package_manager == "zypper":
            commands.append(("Node.js LTS", 80, "sudo zypper --non-interactive install nodejs npm"))

    if "ollama" in missing_tools:
        commands.append(("Ollama", 1200, "curl -fsSL https://ollama.com/install.sh | sh"))

    notes.append("For Linux, Ollama may require relogin/service restart after installation.")
    return commands, notes


def _run_install_command(command: Any) -> bool:
    try:
        if isinstance(command, list):
            result = subprocess.run(command, check=False)
        else:
            result = subprocess.run(str(command), shell=True, check=False)
    except Exception as exc:
        print(f"Install command failed to start: {exc}")
        return False
    return result.returncode == 0


def check_external_tools() -> bool:
    status = _collect_dependency_status()
    required_tools = ("ffmpeg", "ffprobe", "ollama")
    missing_required = [tool for tool in required_tools if not status.get(tool)]
    missing_optional = [tool for tool in ("node",) if not status.get(tool)]

    print("\nDependency doctor:")
    for key in ("python", "ffmpeg", "ffprobe", "ollama", "node", "deno", "yt_dlp_module"):
        state = "OK" if status.get(key) else "MISSING"
        print(f"- {key}: {state}")

    probe_ok, probe_message = _probe_youtube_cookie_sources()
    if probe_ok:
        print(f"- youtube_cookie_probe: OK ({probe_message})")
    else:
        print(f"- youtube_cookie_probe: WARNING ({probe_message})")

    if missing_optional:
        print(f"Optional tools missing: {', '.join(missing_optional)}")

    if not missing_required:
        return True

    missing_tools = sorted(set(missing_required))
    commands, notes = _build_install_commands(missing_tools)
    if commands:
        total_estimate = sum(item[1] for item in commands)
        print("\nInstall plan:")
        for label, estimate_mb, command in commands:
            command_text = " ".join(command) if isinstance(command, list) else str(command)
            print(f"- {label}: ~{estimate_mb} MB")
            print(f"  Command: {command_text}")
        print(f"Estimated total download size: ~{total_estimate} MB")
    else:
        print("\nNo automatic installer is available for the current platform setup.")

    for note in notes:
        print(f"Note: {note}")

    if not commands:
        return len(missing_required) == 0

    if not sys.stdin.isatty():
        print("Non-interactive shell detected; skipping auto-install prompt.")
        return len(missing_required) == 0

    reply = input("Install missing dependencies now? [y/N]: ").strip().lower()
    if reply not in {"y", "yes"}:
        print("Auto-install skipped by user.")
        return len(missing_required) == 0

    all_ok = True
    for label, _, command in commands:
        print(f"Installing {label}...")
        ok = _run_install_command(command)
        all_ok = all_ok and ok
        if not ok:
            print(f"Warning: install command failed for {label}.")

    post_status = _collect_dependency_status()
    post_missing_required = [tool for tool in required_tools if not post_status.get(tool)]
    if post_missing_required:
        print(f"Required tools still missing: {', '.join(post_missing_required)}")
        return False
    if not all_ok:
        print("Some optional installs failed; continuing.")
    return True


def should_skip_setup(
    state: Dict[str, Any],
    req_hash: str,
    requested_torch: str,
    resolved_torch_target: str,
    torch_info: Dict[str, Any],
    force: bool,
) -> bool:
    if force:
        return False
    if state.get("requirements_hash") != req_hash:
        return False
    if state.get("python_executable") != sys.executable:
        return False
    if state.get("torch_request") != requested_torch:
        return False

    installed_mode = str(state.get("torch_installed_mode") or "").lower().strip()
    if requested_torch == "auto" and installed_mode in {"cpu", "cuda", "skip"}:
        return torch_satisfies_target(torch_info, installed_mode)

    if str(state.get("torch_target") or "").lower().strip() != resolved_torch_target:
        return False
    return torch_satisfies_target(torch_info, resolved_torch_target)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Python environment for AI Auto Clipper.")
    parser.add_argument(
        "--torch",
        choices=("auto", "cuda", "cpu", "skip"),
        default="auto",
        help="Torch installation mode. Default: auto",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reinstall of dependencies even if setup state matches.",
    )
    parser.add_argument(
        "--skip-pip-upgrade",
        action="store_true",
        help="Skip pip self-upgrade step.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not REQUIREMENTS_FILE.exists():
        print(f"Error: requirements file not found: {REQUIREMENTS_FILE}")
        return 1

    if not running_in_venv():
        print("Warning: no virtual environment detected. Continuing anyway.")

    has_nvidia_gpu = detect_nvidia_gpu()
    torch_target = resolve_torch_target(args.torch, has_nvidia_gpu)
    req_hash = requirements_hash()
    setup_state = read_setup_state()
    torch_info_before = probe_torch()

    if should_skip_setup(
        setup_state,
        req_hash,
        args.torch,
        torch_target,
        torch_info_before,
        args.force,
    ):
        print("Environment already prepared. Skipping dependency install.")
        tools_ok = check_external_tools()
        return 0 if tools_ok else 1

    try:
        if not args.skip_pip_upgrade:
            print("Upgrading pip...")
            pip_install(["install", "--upgrade", "pip"])

        print("Installing project requirements...")
        pip_install(["install", "-r", str(REQUIREMENTS_FILE)])

        resolved_torch_mode, torch_info_after = install_torch(torch_target)
    except subprocess.CalledProcessError as exc:
        print(f"Dependency installation failed. Command: {' '.join(exc.cmd)}")
        return exc.returncode or 1

    setup_payload = {
        "python_executable": sys.executable,
        "requirements_hash": req_hash,
        "torch_request": args.torch,
        "torch_target": torch_target,
        "torch_resolved_target": torch_target,
        "torch_installed_mode": resolved_torch_mode,
        "torch_version": torch_info_after.get("version"),
        "torch_cuda_available": torch_info_after.get("cuda_available"),
        "torch_cuda_version": torch_info_after.get("cuda_version"),
    }
    write_setup_state(setup_payload)

    print("Environment setup complete.")
    if torch_info_after.get("installed"):
        print(
            f"Torch version: {torch_info_after.get('version')} "
            f"(CUDA available: {torch_info_after.get('cuda_available')})"
        )
    tools_ok = check_external_tools()
    return 0 if tools_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
