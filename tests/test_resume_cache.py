from __future__ import annotations

import io
import logging
import sys
import tempfile
import types
import unittest
from pathlib import Path

from rich.console import Console

sys.modules.setdefault("yt_dlp", types.SimpleNamespace())

from core.engine import ClippingEngine
from utils.validators import build_default_config


class ResumeCacheTests(unittest.TestCase):
    def _engine(self, base_dir: Path) -> ClippingEngine:
        logger = logging.getLogger(f"resume-cache-test-{id(base_dir)}")
        logger.handlers.clear()
        logger.addHandler(logging.NullHandler())
        config = build_default_config()
        console = Console(file=io.StringIO(), force_terminal=False)
        return ClippingEngine(base_dir=base_dir, config=config, logger=logger, console=console)

    def test_in_progress_checkpoint_loads_for_same_video_and_settings(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = self._engine(Path(temp_dir))
            video = engine.input_dir / "sample.mp4"
            video.write_bytes(b"fake video bytes")
            signature = engine._current_resume_signature()

            checkpoint = {
                "config_signature": signature,
                "transcript": [[0.0, 1.0, "hello"]],
                "transcript_meta": engine._current_transcript_meta(),
            }
            engine._save_video_checkpoint(video, checkpoint)

            loaded = engine._load_video_checkpoint(video, signature)

            self.assertEqual(loaded["transcript"], [[0.0, 1.0, "hello"]])
            self.assertEqual(loaded["config_signature"], signature)
            self.assertTrue(loaded["video_fingerprint"])

    def test_completed_checkpoint_is_removed_so_reruns_start_clean(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = self._engine(Path(temp_dir))
            video = engine.input_dir / "sample.mp4"
            video.write_bytes(b"fake video bytes")
            signature = engine._current_resume_signature()

            checkpoint = {
                "config_signature": signature,
                "transcript": [[0.0, 1.0, "hello"]],
                "transcript_meta": engine._current_transcript_meta(),
                "completed": True,
            }
            engine._save_video_checkpoint(video, checkpoint)
            checkpoint_path = engine._video_checkpoint_path(video)
            self.assertTrue(checkpoint_path.exists())

            loaded = engine._load_video_checkpoint(video, signature)

            self.assertEqual(loaded, {})
            self.assertFalse(checkpoint_path.exists())

    def test_checkpoint_can_be_cleared_after_source_is_archived(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = self._engine(Path(temp_dir))
            video = engine.input_dir / "sample.mp4"
            video.write_bytes(b"fake video bytes")
            signature = engine._current_resume_signature()

            checkpoint = {"config_signature": signature}
            engine._save_video_checkpoint(video, checkpoint)
            fingerprint = str(checkpoint["video_fingerprint"])
            checkpoint_path = engine._video_checkpoint_path(video, fingerprint=fingerprint)
            video.unlink()

            engine._clear_video_checkpoint(video, fingerprint=fingerprint)

            self.assertFalse(checkpoint_path.exists())


if __name__ == "__main__":
    unittest.main()
