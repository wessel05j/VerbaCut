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
from utils.hardware_detect import CudaMemorySnapshot


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

    def _engine_for_cuda_context(self) -> ClippingEngine:
        engine = object.__new__(ClippingEngine)
        engine.config = {
            "ollama": {"url": "http://localhost:11434"},
            "runtime": {"adaptive_cuda_context": True},
        }
        return engine

    def test_cuda_context_uses_configured_maximum_with_enough_headroom(self) -> None:
        engine = self._engine_for_cuda_context()
        snapshot = CudaMemorySnapshot(
            device_index=0,
            device_name="RTX Test",
            total_gb=24.0,
            free_gb=22.0,
            used_gb=2.0,
            source="test",
        )

        with (
            mock.patch("core.engine.detect_cuda_memory_snapshot", return_value=snapshot),
            mock.patch.object(engine, "_loaded_ollama_model_vram_gb", return_value=0.0),
        ):
            effective, details = engine._resolve_runtime_context_window(
                model_name="gpt-oss:20b",
                configured_context_tokens=16384,
            )

        self.assertEqual(effective, 16384)
        self.assertEqual(details["context_headroom_gb"], 9.6)
        self.assertFalse(details["model_resident"])

    def test_cuda_context_scales_down_when_model_has_little_projected_headroom(self) -> None:
        engine = self._engine_for_cuda_context()
        snapshot = CudaMemorySnapshot(
            device_index=0,
            device_name="RTX Test",
            total_gb=16.0,
            free_gb=13.0,
            used_gb=3.0,
            source="test",
        )

        with (
            mock.patch("core.engine.detect_cuda_memory_snapshot", return_value=snapshot),
            mock.patch.object(engine, "_loaded_ollama_model_vram_gb", return_value=0.0),
        ):
            effective, details = engine._resolve_runtime_context_window(
                model_name="gpt-oss:20b",
                configured_context_tokens=16384,
            )

        self.assertEqual(effective, 8192)
        self.assertAlmostEqual(details["context_headroom_gb"], 0.6)
        self.assertEqual(details["scale"], 0.5)

    def test_loaded_model_is_not_reserved_twice(self) -> None:
        engine = self._engine_for_cuda_context()
        snapshot = CudaMemorySnapshot(
            device_index=0,
            device_name="RTX Test",
            total_gb=16.0,
            free_gb=3.0,
            used_gb=13.0,
            source="test",
        )

        with (
            mock.patch("core.engine.detect_cuda_memory_snapshot", return_value=snapshot),
            mock.patch.object(engine, "_loaded_ollama_model_vram_gb", return_value=12.0),
        ):
            effective, details = engine._resolve_runtime_context_window(
                model_name="gpt-oss:20b",
                configured_context_tokens=16384,
            )

        self.assertEqual(effective, 14336)
        self.assertEqual(details["context_headroom_gb"], 3.0)
        self.assertTrue(details["model_resident"])

    def test_missing_cuda_telemetry_preserves_configured_context(self) -> None:
        engine = self._engine_for_cuda_context()

        with mock.patch("core.engine.detect_cuda_memory_snapshot", return_value=None):
            effective, details = engine._resolve_runtime_context_window(
                model_name="gpt-oss:20b",
                configured_context_tokens=16384,
            )

        self.assertEqual(effective, 16384)
        self.assertFalse(details["detected"])
        self.assertEqual(details["reason"], "cuda-telemetry-unavailable")


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
