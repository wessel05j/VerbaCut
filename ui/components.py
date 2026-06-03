from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from utils.validators import APP_GITHUB, APP_NAME, APP_VERSION


ASCII_LOGO = r"""
__     __        _           ____      _
\ \   / /__ _ __| |__   __ _/ ___|   _| |_
 \ \ / / _ \ '__| '_ \ / _` | |  | | | | __|
  \ V /  __/ |  | |_) | (_| | |__| |_| | |_
   \_/ \___|_|  |_.__/ \__,_|\____\__,_|\__|
"""


def build_header_panel() -> Panel:
    title = Text(ASCII_LOGO, style="bold cyan")
    subtitle = Text(f"{APP_NAME.upper()} v{APP_VERSION}", style="bold white")
    github = Text(APP_GITHUB, style="bright_blue")
    body = Align.center(Text.assemble(title, "\n", subtitle, "\n", github))
    return Panel(body, border_style="cyan", padding=(1, 2))


def show_header(console: Console) -> None:
    console.print(build_header_panel())


def hard_clear(console: Console) -> None:
    """
    Clear visible terminal and request scrollback purge (best-effort).
    """
    if os.name == "nt":
        try:
            os.system("cls")
        except Exception:
            pass
    if sys.stdout.isatty():
        try:
            # 3J clears scrollback on terminals that support ANSI VT sequences.
            sys.stdout.write("\033[3J\033[2J\033[H")
            sys.stdout.flush()
        except Exception:
            pass
    console.clear(home=True)


def startup_animation(console: Console) -> None:
    steps = [
        "Loading terminal UI",
        "Checking local runtime",
        "Preparing AI components",
    ]
    with console.status("[bold cyan]Booting VerbaCut...", spinner="dots"):
        for _ in steps:
            time.sleep(0.22)


def warning_panel(message: str) -> Panel:
    return Panel(
        Text(message, style="bold yellow"),
        title="Warning",
        border_style="yellow",
    )


def success_panel(message: str) -> Panel:
    return Panel(
        Text(message, style="bold green"),
        title="Ready",
        border_style="green",
    )


def _command_is_available(executable: str) -> bool:
    command_path = Path(executable)
    if command_path.is_file():
        return True
    return shutil.which(executable) is not None


def _normalized_command_name(executable: str) -> str:
    name = Path(executable).name.lower()
    for suffix in (".exe", ".cmd", ".bat", ".com"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _coalesce_windows_executable(parts: list[str]) -> list[str]:
    if os.name != "nt" or len(parts) < 2 or _command_is_available(parts[0]):
        return parts
    executable_suffixes = (".exe", ".cmd", ".bat", ".com")
    for index in range(2, len(parts) + 1):
        candidate = " ".join(parts[:index])
        if _command_is_available(candidate):
            return [candidate, *parts[index:]]
        if parts[index - 1].lower().endswith(executable_suffixes):
            return [candidate, *parts[index:]]
    return parts


def _split_editor_command(raw_command: str) -> list[str]:
    if not raw_command:
        return []
    try:
        parts = shlex.split(raw_command, posix=os.name != "nt")
    except ValueError:
        parts = raw_command.split()
    cleaned = [part.strip() for part in parts if part.strip()]
    return _coalesce_windows_executable(cleaned)


def _ensure_editor_wait_flag(parts: list[str]) -> list[str]:
    if not parts:
        return parts
    command_name = _normalized_command_name(parts[0])
    lowered_args = {arg.lower() for arg in parts[1:]}
    if command_name in {"code", "code-insiders", "codium"} and "--wait" not in lowered_args:
        return [parts[0], "--wait", *parts[1:]]
    if command_name in {"subl", "sublime_text"} and "-w" not in lowered_args and "--wait" not in lowered_args:
        return [parts[0], "-w", *parts[1:]]
    return parts


def _editor_commands(temp_file: Path) -> list[list[str]]:
    target = str(temp_file)
    commands: list[list[str]] = []
    visual = os.environ.get("VISUAL", "").strip()
    editor = os.environ.get("EDITOR", "").strip()
    for candidate in (visual, editor):
        parts = _split_editor_command(candidate)
        if not parts:
            continue
        parts = _ensure_editor_wait_flag(parts)
        if not _command_is_available(parts[0]):
            continue
        commands.append([*parts, target])

    if os.name == "nt":
        commands.append(["notepad", target])
        if shutil.which("code"):
            commands.append(["code", "--wait", target])
    elif sys.platform == "darwin":
        if shutil.which("nano"):
            commands.append(["nano", target])
        elif shutil.which("vi"):
            commands.append(["vi", target])
        else:
            commands.append(["open", "-W", "-t", target])
    else:
        if shutil.which("nano"):
            commands.append(["nano", target])
        else:
            commands.append(["vi", target])

    unique_commands: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for command in commands:
        key = tuple(command)
        if key in seen:
            continue
        seen.add(key)
        unique_commands.append(command)
    return unique_commands


def edit_text_in_editor(initial_text: str, filename_hint: str = "system_prompt.txt") -> str:
    """
    Open text in external editor for full navigation/editing.
    Falls back to returning the original text when editor launch fails.
    """
    temp_dir = Path(tempfile.gettempdir())
    temp_dir.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=f"_{filename_hint}",
        encoding="utf-8",
        delete=False,
        dir=str(temp_dir),
    )
    try:
        handle.write(initial_text)
        handle.flush()
        temp_file = Path(handle.name)
    finally:
        handle.close()

    editor_ran = False
    for command in _editor_commands(temp_file):
        try:
            result = subprocess.run(command, check=False)
            if result.returncode not in (0, None):
                continue
            editor_ran = True
            break
        except Exception:
            continue

    if not editor_ran:
        try:
            temp_file.unlink(missing_ok=True)
        except Exception:
            pass
        return initial_text

    try:
        edited = temp_file.read_text(encoding="utf-8")
    except Exception:
        return initial_text
    finally:
        try:
            temp_file.unlink(missing_ok=True)
        except Exception:
            pass

    cleaned = edited.strip()
    if not cleaned:
        return initial_text
    return edited
