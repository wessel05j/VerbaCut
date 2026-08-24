from __future__ import annotations

import re
import sys
import types
from pathlib import Path
from unittest import TestCase, mock

sys.modules.setdefault("yt_dlp", types.SimpleNamespace(YoutubeDL=object))

from core.ai_pipeline import chunk_transcript
from core.clipping import transcribe_video
from core.engine import ClippingEngine


class _FakeWhisperModel:
    def __init__(self) -> None:
        self.calls = []

    def transcribe(self, path: str, verbose: bool = False, word_timestamps: bool = False):
        self.calls.append(
            {
                "path": path,
                "verbose": verbose,
                "word_timestamps": word_timestamps,
            }
        )
        return {
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "Hello",
                    "words": [
                        {"start": 0.0, "end": 0.45, "word": " Hello"},
                    ],
                },
                {
                    "start": 1.1,
                    "end": 2.0,
                    "text": "world.",
                    "words": [
                        {"start": 1.1, "end": 1.7, "word": " world."},
                    ],
                },
            ]
        }


class TimedChunkingTests(TestCase):
    def _fake_estimate_tokens(self, text: str) -> int:
        return len(re.findall(r"[A-Za-z]+", str(text))) * 100

    def test_oversized_segment_uses_real_word_ranges(self) -> None:
        transcript = [
            [
                0.0,
                8.0,
                "alpha beta gamma delta epsilon zeta eta theta",
                [
                    [0.0, 1.0, " alpha"],
                    [1.0, 2.0, " beta"],
                    [2.0, 3.0, " gamma"],
                    [3.0, 4.0, " delta"],
                    [4.0, 5.0, " epsilon"],
                    [5.0, 6.0, " zeta"],
                    [6.0, 7.0, " eta"],
                    [7.0, 8.0, " theta"],
                ],
            ]
        ]

        with mock.patch("core.ai_pipeline.estimate_tokens", side_effect=self._fake_estimate_tokens):
            chunks = chunk_transcript(transcript=transcript, max_tokens=300, safety_margin=0)

        flattened = [segment for chunk in chunks for segment in chunk]
        self.assertEqual(
            flattened,
            [
                [0.0, 3.0, "alpha beta gamma"],
                [3.0, 6.0, "delta epsilon zeta"],
                [6.0, 8.0, "eta theta"],
            ],
        )

    def test_hardware_pressure_does_not_reduce_transcript_context(self) -> None:
        engine = object.__new__(ClippingEngine)
        engine.hardware_profile = {"gpu_vram_gb": 4.0, "ram_gb": 32.0}
        engine.config = {"runtime": {"setup_intensity": "balanced"}}

        effective, pressure_ratio = engine._resolve_runtime_chunk_cap(
            model_name="gpt-oss:20b",
            configured_chunk_tokens=6000,
        )

        self.assertEqual(effective, 6000)
        self.assertGreater(pressure_ratio, 1.0)


class TranscriptionTests(TestCase):
    def test_transcription_keeps_word_timestamps_for_merged_segments(self) -> None:
        whisper_model = _FakeWhisperModel()

        transcript = transcribe_video(Path("sample.mp4"), whisper_model, min_pause=3.0)

        self.assertEqual(len(transcript), 1)
        self.assertEqual(
            transcript[0],
            [
                0.0,
                2.0,
                "Hello world.",
                [
                    [0.0, 0.45, " Hello"],
                    [1.1, 1.7, " world."],
                ],
            ],
        )
        self.assertEqual(len(whisper_model.calls), 1)
        self.assertTrue(whisper_model.calls[0]["word_timestamps"])


class CheckpointCompatibilityTests(TestCase):
    def test_transcript_checkpoint_requires_current_word_timestamp_schema(self) -> None:
        self.assertTrue(ClippingEngine._transcript_checkpoint_compatible({}))
        self.assertFalse(ClippingEngine._transcript_checkpoint_compatible({"transcript": []}))
        self.assertFalse(
            ClippingEngine._transcript_checkpoint_compatible(
                {
                    "transcript": [],
                    "transcript_meta": {"schema_version": 1, "word_timestamps": True},
                }
            )
        )
        self.assertFalse(
            ClippingEngine._transcript_checkpoint_compatible(
                {
                    "transcript": [],
                    "transcript_meta": {"schema_version": 2, "word_timestamps": False},
                }
            )
        )
        self.assertTrue(
            ClippingEngine._transcript_checkpoint_compatible(
                {
                    "transcript": [],
                    "transcript_meta": {"schema_version": 2, "word_timestamps": True},
                }
            )
        )
