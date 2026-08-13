from __future__ import annotations

import logging
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

from automation_worker import _pid_is_running, _worker_lock
from core.engine import ClippingEngine
from core.yt_handler import YTHandler
from utils.validators import build_default_config, validate_config


class AutomationConfigTests(unittest.TestCase):
    def test_720_hour_fetch_window_is_valid(self) -> None:
        config = build_default_config()
        config["clipping"]["channels_hours_limit"] = 720

        self.assertEqual(validate_config(config), [])

    def test_automation_engine_uses_its_in_memory_signature(self) -> None:
        engine = object.__new__(ClippingEngine)
        engine.persist_config = False
        engine.config = build_default_config()

        self.assertEqual(engine._disk_resume_signature(), engine._current_resume_signature())

    def test_download_handoff_records_summary_and_files(self) -> None:
        from automation_worker import AutomationEngine

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            state_file = root / "state.json"
            input_dir = root / "input"
            input_dir.mkdir()
            (input_dir / "source.mp4").write_bytes(b"video")
            engine = object.__new__(AutomationEngine)
            engine.input_dir = input_dir
            engine._automation_state_file = state_file
            engine._automation_run_state = {"run_id": "test"}

            engine._after_downloads({"downloaded": 1, "failed": 0})

            state = __import__("json").loads(state_file.read_text(encoding="utf-8"))
            self.assertEqual(state["stage"], "downloads_complete")
            self.assertEqual(state["download_summary"]["downloaded"], 1)
            self.assertEqual(len(state["downloaded_files"]), 1)


class AutomationLockTests(unittest.TestCase):
    def test_worker_lock_is_removed_after_success(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            lock_path = Path(temp_dir) / "worker.lock.json"
            with _worker_lock(lock_path, "test-run"):
                self.assertTrue(lock_path.exists())
            self.assertFalse(lock_path.exists())

    def test_pid_probe_reports_current_worker(self) -> None:
        import os

        self.assertTrue(_pid_is_running(os.getpid()))


class RecentChannelFallbackTests(unittest.TestCase):
    def test_detailed_fallback_completes_window_beyond_rss_limit(self) -> None:
        class FakeResponse:
            status_code = 200
            text = "".join(
                f"<entry><yt:videoId>AAAAAAAAA{i:02d}</yt:videoId>"
                f"<published>2026-08-{13 - (i // 3):02d}T01:00:00+00:00</published></entry>"
                for i in range(15)
            )

        detailed_entries = [
            {
                "id": f"BBBBBBBBB{i:02d}",
                "url": f"https://www.youtube.com/watch?v=BBBBBBBBB{i:02d}",
                "timestamp": datetime(2026, 8, 8, 1, 0).timestamp(),
            }
            for i in range(5)
        ]
        handler = YTHandler(Path(tempfile.mkdtemp()), logging.getLogger("recent-fallback"))
        handler._apply_cookies_to_opts = mock.Mock()

        outer = mock.MagicMock()
        outer.extract_info.return_value = {"channel_id": "UCAuk798iHprjTtwlClkFxMA"}
        detailed = mock.MagicMock()
        detailed.extract_info.return_value = {"entries": detailed_entries}
        managers = [
            mock.MagicMock(__enter__=mock.Mock(return_value=outer), __exit__=mock.Mock(return_value=False)),
            mock.MagicMock(__enter__=mock.Mock(return_value=detailed), __exit__=mock.Mock(return_value=False)),
        ]

        with mock.patch("core.yt_handler.datetime") as fake_datetime, mock.patch(
            "core.yt_handler.requests.get", return_value=FakeResponse()
        ), mock.patch("core.yt_handler.yt_dlp.YoutubeDL", side_effect=managers):
            fake_datetime.now.return_value = datetime(2026, 8, 13, 12, 0).astimezone()
            fake_datetime.fromisoformat.side_effect = datetime.fromisoformat
            fake_datetime.fromtimestamp.side_effect = datetime.fromtimestamp
            links = handler.list_recent_channel_links(
                ["https://www.youtube.com/@sam_sulek/videos"], hours_limit=720, playlistend=20
            )

        self.assertEqual(len(links), 20)


if __name__ == "__main__":
    unittest.main()
