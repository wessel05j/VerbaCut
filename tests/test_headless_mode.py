from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rich.console import Console

from core.engine import ClippingEngine
from core.format_checker import FormatDecision, QUALITY_1080, choose_download_format
from core.yt_handler import YTHandler


class HeadlessEngineTests(unittest.TestCase):
    def test_download_warning_does_not_prompt_in_headless_mode(self) -> None:
        engine = object.__new__(ClippingEngine)
        engine.console = Console(quiet=True)
        engine.logger = logging.getLogger("test-headless-engine")
        engine.interactive = False

        with mock.patch("builtins.input") as prompt:
            engine._show_youtube_download_issues(
                {
                    "failed": 1,
                    "invalid": 0,
                    "issues": [{"url": "https://example.test/video", "error": "failed"}],
                }
            )

        prompt.assert_not_called()

    def test_config_persistence_can_be_disabled(self) -> None:
        engine = object.__new__(ClippingEngine)
        engine.persist_config = False
        engine.config = {"clipping": {"youtube_links": []}}
        engine._config_paths = {"config_file": Path("unused.json")}

        with mock.patch("core.engine.save_json_file") as save:
            engine._persist_config()

        save.assert_not_called()


class FormatPreflightTests(unittest.TestCase):
    def test_download_selection_can_cap_4k_inventory_at_1080(self) -> None:
        decision = choose_download_format(
            {
                "formats": [
                    {
                        "format_id": "313",
                        "width": 3840,
                        "height": 2160,
                        "vcodec": "vp9",
                        "acodec": "none",
                        "tbr": 8000,
                    },
                    {
                        "format_id": "137",
                        "width": 1920,
                        "height": 1080,
                        "vcodec": "avc1",
                        "acodec": "none",
                        "tbr": 2000,
                    },
                ]
            },
            maximum_quality_rank=QUALITY_1080,
        )

        self.assertTrue(decision.acceptable)
        self.assertEqual(decision.format_id, "137")
        self.assertEqual(decision.quality_rank, QUALITY_1080)

    def test_preflight_falls_back_from_cookie_probe_to_no_cookie_probe(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            handler = YTHandler(base_dir=Path(temp_dir), logger=logging.getLogger("test-format-probe"))
            good = FormatDecision(
                format_string="137+bestaudio",
                strategy="1080-class-video-plus-audio",
                acceptable=True,
                quality_rank=QUALITY_1080,
                quality_label="1080-class",
                format_id="137",
                width=1920,
                height=1080,
            )
            handler._download_strategies = mock.Mock(
                return_value=[
                    {"name": "cookies-generic", "use_cookies": True},
                    {"name": "no-cookies-generic", "use_cookies": False},
                ]
            )
            handler._configure_strategy_opts = mock.Mock(side_effect=lambda opts, strategy: opts)
            handler._probe_and_decide_format = mock.Mock(
                side_effect=[RuntimeError("cookie probe failed"), good]
            )

            result = handler.probe_download_format("https://www.youtube.com/watch?v=QUP4hRbcNFc")

        self.assertTrue(result.acceptable)
        self.assertEqual(result.strategy, "no-cookies-generic")
        self.assertEqual(result.decision, good)


if __name__ == "__main__":
    unittest.main()
