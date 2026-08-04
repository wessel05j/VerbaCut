from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import re
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List

import yt_dlp.version
from rich.console import Console

from core.engine import ClippingEngine
from core.format_checker import QUALITY_1080
from core.yt_handler import YTHandler
from utils.logging_setup import setup_logging
from utils.validators import ensure_runtime_layout, load_validated_config


MINIMUM_YT_DLP_VERSION = (2026, 7, 4)


def _version_key(value: str) -> tuple[int, int, int]:
    parts = [int(part) for part in re.findall(r"\d+", str(value))[:3]]
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])


def _load_links(path: Path) -> List[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Links file does not exist: {path}")

    links: List[str] = []
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "source_url" not in reader.fieldnames:
                raise ValueError("CSV links file must contain a source_url column.")
            links = [str(row.get("source_url", "")).strip() for row in reader]
    else:
        links = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()]

    unique: List[str] = []
    seen = set()
    for link in links:
        if not link or link.startswith("#") or link in seen:
            continue
        seen.add(link)
        unique.append(link)
    if not unique:
        raise ValueError("Links file contains no usable URLs.")
    return unique


def _write_result(workspace: Path, payload: Dict[str, Any]) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    payload["written_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    (workspace / "headless_result.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


@contextmanager
def _headless_console(workspace: Path) -> Iterator[Console]:
    """Use a regular UTF-8 file instead of Rich's legacy Windows console API."""

    log_path = workspace / "engine_console.log"
    with log_path.open("a", encoding="utf-8", newline="\n") as stream:
        console = Console(
            file=stream,
            force_terminal=False,
            color_system=None,
            no_color=True,
            width=120,
        )
        yield console


def _enable_workspace_diagnostics(config: Dict[str, Any], workspace: Path) -> None:
    """Persist transcripts and exact clip records before resumable cache cleanup."""

    diagnostics = config.setdefault("diagnostics", {})
    if not isinstance(diagnostics, dict):
        diagnostics = {}
        config["diagnostics"] = diagnostics
    diagnostics["enabled"] = True
    diagnostics["artifacts_dir"] = str(workspace / "system" / "diagnostics")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run VerbaCut without its interactive dashboard.")
    parser.add_argument("--links-file", required=True, type=Path)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--query-file", type=Path)
    parser.add_argument("--minimum-sources", type=int, default=1)
    parser.add_argument("--minimum-exports", type=int, default=1)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    base_dir = Path(__file__).resolve().parent
    workspace = args.workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    installed_version = str(yt_dlp.version.__version__)
    if _version_key(installed_version) < MINIMUM_YT_DLP_VERSION:
        _write_result(
            workspace,
            {
                "status": "blocked",
                "stage": "dependency_preflight",
                "error": (
                    f"yt-dlp {installed_version} is below required version "
                    f"{'.'.join(str(part) for part in MINIMUM_YT_DLP_VERSION)}."
                ),
            },
        )
        return 2

    ensure_runtime_layout(base_dir)
    valid, loaded, errors = load_validated_config(base_dir)
    if not valid or loaded is None:
        _write_result(
            workspace,
            {"status": "blocked", "stage": "config_preflight", "error": "; ".join(errors)},
        )
        return 2

    links = _load_links(args.links_file.resolve())
    logger = setup_logging(
        base_dir=base_dir,
        log_level=str(loaded["app"].get("log_level", "INFO")),
        dev_mode=bool(loaded["app"].get("dev_mode", False)),
    )
    probe_handler = YTHandler(
        base_dir=base_dir,
        logger=logger,
        maximum_quality_rank=QUALITY_1080,
    )
    accepted: List[str] = []
    probes: List[Dict[str, Any]] = []
    for link in links:
        result = probe_handler.probe_download_format(link)
        decision = result.decision
        probes.append(
            {
                "url": link,
                "acceptable": result.acceptable,
                "strategy": result.strategy,
                "quality": decision.quality_label if decision else None,
                "format_id": decision.format_id if decision else None,
                "width": decision.width if decision else 0,
                "height": decision.height if decision else 0,
                "error": result.error,
            }
        )
        if result.acceptable:
            accepted.append(link)

    minimum_sources = max(1, int(args.minimum_sources))
    if len(accepted) < minimum_sources:
        _write_result(
            workspace,
            {
                "status": "blocked",
                "stage": "format_preflight",
                "yt_dlp_version": installed_version,
                "required_sources": minimum_sources,
                "accepted_sources": len(accepted),
                "probes": probes,
            },
        )
        return 2

    if args.preflight_only:
        _write_result(
            workspace,
            {
                "status": "ready",
                "stage": "format_preflight",
                "yt_dlp_version": installed_version,
                "accepted_sources": len(accepted),
                "probes": probes,
            },
        )
        return 0

    config = copy.deepcopy(loaded)
    config["paths"]["input_dir"] = str(workspace / "input")
    config["paths"]["output_dir"] = str(workspace / "exports")
    config["paths"]["temp_dir"] = str(workspace / "temp")
    config["paths"]["system_dir"] = str(workspace / "system")
    config["clipping"]["youtube_links"] = accepted
    config["clipping"]["max_download_quality_rank"] = QUALITY_1080
    _enable_workspace_diagnostics(config, workspace)
    if args.query_file:
        config["clipping"]["user_query"] = args.query_file.resolve().read_text(
            encoding="utf-8-sig"
        ).strip()

    try:
        with _headless_console(workspace) as console:
            engine = ClippingEngine(
                base_dir=base_dir,
                config=config,
                logger=logger,
                console=console,
                interactive=False,
                persist_config=False,
            )
            engine.run()
    except Exception as exc:
        logger.exception("Headless VerbaCut run failed")
        _write_result(
            workspace,
            {
                "status": "failed",
                "stage": "engine",
                "yt_dlp_version": installed_version,
                "probes": probes,
                "error": str(exc),
                "engine_console_log": str((workspace / "engine_console.log").resolve()),
                "diagnostics_dir": str((workspace / "system" / "diagnostics").resolve()),
            },
        )
        return 3

    export_dir = workspace / "exports"
    exports = sorted(
        str(path.resolve())
        for pattern in ("*.mp4", "*.mkv", "*.webm")
        for path in export_dir.glob(pattern)
        if path.is_file()
    )
    minimum_exports = max(1, int(args.minimum_exports))
    status = "complete" if len(exports) >= minimum_exports else "insufficient_exports"
    _write_result(
        workspace,
        {
            "status": status,
            "stage": "complete",
            "yt_dlp_version": installed_version,
            "accepted_sources": len(accepted),
            "probes": probes,
            "minimum_exports": minimum_exports,
            "exports": exports,
            "engine_console_log": str((workspace / "engine_console.log").resolve()),
            "diagnostics_dir": str((workspace / "system" / "diagnostics").resolve()),
        },
    )
    return 0 if status == "complete" else 4


if __name__ == "__main__":
    raise SystemExit(main())
