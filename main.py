from __future__ import annotations

from pathlib import Path

from rich.console import Console

from ui.dashboard import Dashboard
from ui.setup_wizard import SetupWizard
from utils.logging_setup import setup_logging
from utils.validators import ensure_runtime_layout, load_validated_config


def main() -> int:
    base_dir = Path(__file__).resolve().parent
    ensure_runtime_layout(base_dir)

    console = Console()
    logger = setup_logging(base_dir=base_dir, log_level="INFO", dev_mode=False)

    is_valid, config, errors = load_validated_config(base_dir)
    if not is_valid or config is None:
        missing_config = bool(errors) and all("not found" in str(error).lower() for error in errors)
        if errors and not missing_config:
            logger.error("Config invalid. Setup preserved; refusing to launch the setup wizard. Issues: %s", "; ".join(errors))
            console.print("[bold red]VerbaCut configuration needs attention.[/bold red]")
            for error in errors:
                console.print(f"[red]- {error}[/red]")
            console.print(
                "[yellow]Your existing config and saved prompts were preserved. "
                "Fix the listed setting or explicitly choose Re-run Setup Wizard from the dashboard.[/yellow]"
            )
            return 2
        elif missing_config:
            logger.info("No config found. Starting setup wizard.")
        wizard = SetupWizard(base_dir=base_dir, console=console, logger=logger)
        try:
            config = wizard.run()
        except Exception as exc:
            logger.error("Setup wizard failed: %s", exc)
            console.print(f"[bold red]Setup failed:[/bold red] {exc}")
            return 1

    logger = setup_logging(
        base_dir=base_dir,
        log_level=str(config["app"].get("log_level", "INFO")),
        dev_mode=bool(config["app"].get("dev_mode", False)),
    )
    dashboard = Dashboard(base_dir=base_dir, config=config, console=console, logger=logger)
    dashboard.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
