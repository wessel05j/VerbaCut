from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


def setup_logging(
    base_dir: Path,
    log_level: str = "INFO",
    dev_mode: bool = False,
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """Configure root logger for file + terminal output."""
    logs_dir = base_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    resolved_log_file = Path(log_file) if log_file is not None else logs_dir / "app.log"
    resolved_log_file.parent.mkdir(parents=True, exist_ok=True)

    effective_level_name = "DEBUG" if dev_mode else log_level.upper()
    effective_level = getattr(logging, effective_level_name, logging.INFO)

    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    root_logger.setLevel(effective_level)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(
        filename=resolved_log_file,
        maxBytes=2_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(effective_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.ERROR if not dev_mode else logging.DEBUG)
    console_handler.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
    root_logger.addHandler(console_handler)

    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("yt_dlp").setLevel(logging.WARNING)
    logging.getLogger("moviepy").setLevel(logging.WARNING)
    logging.getLogger("py.warnings").setLevel(logging.ERROR)
    return logging.getLogger("verbacut")
