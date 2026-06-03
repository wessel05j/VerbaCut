from __future__ import annotations

import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from utils.validators import load_json_file, save_json_file


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
    cleaned = cleaned.strip("._")
    return cleaned or "video"


def _coerce_clip_list(clips: Any) -> List[List[float]]:
    normalized: List[List[float]] = []
    if not isinstance(clips, list):
        return normalized
    for item in clips:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        try:
            start = float(item[0])
            end = float(item[1])
            score = float(item[2]) if len(item) > 2 else 0.0
        except (TypeError, ValueError):
            continue
        normalized.append([start, end, score])
    return normalized


def _coerce_transcript_segments(transcript: Any) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    if not isinstance(transcript, list):
        return normalized
    for index, item in enumerate(transcript):
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        try:
            start = float(item[0])
            end = float(item[1])
        except (TypeError, ValueError):
            continue
        text = str(item[2]).strip()
        word_timestamps = bool(len(item) >= 4 and isinstance(item[3], list))
        normalized.append(
            {
                "index": index,
                "start": start,
                "end": end,
                "duration": max(0.0, end - start),
                "text": text,
                "word_count": len(text.split()),
                "word_timestamps": word_timestamps,
            }
        )
    return normalized


def _probe_duration_seconds(path: Path) -> Optional[float]:
    ffprobe_bin = shutil.which("ffprobe")
    if ffprobe_bin is None or not path.exists():
        return None

    command = [
        ffprobe_bin,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            errors="replace",
            check=False,
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    try:
        duration = float((completed.stdout or "").strip())
    except (TypeError, ValueError):
        return None
    if duration <= 0:
        return None
    return duration


def _duration_tolerance_seconds(expected_duration: float) -> float:
    return max(0.35, min(1.0, max(0.0, float(expected_duration)) * 0.05))


def _load_optional_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return load_json_file(path)
    except Exception:
        return default


def _segment_for_start(transcript: Sequence[Dict[str, Any]], clip_start: float) -> Optional[Dict[str, Any]]:
    for segment in transcript:
        if float(segment["end"]) > clip_start + 1e-6:
            return segment
    if transcript:
        return transcript[-1]
    return None


def _segment_for_end(transcript: Sequence[Dict[str, Any]], clip_end: float) -> Optional[Dict[str, Any]]:
    for segment in reversed(list(transcript)):
        if float(segment["start"]) < clip_end - 1e-6:
            return segment
    if transcript:
        return transcript[0]
    return None


def _start_snippet(transcript: Sequence[Dict[str, Any]], clip_start: float, clip_end: float) -> str:
    selected: List[str] = []
    for segment in transcript:
        if float(segment["end"]) <= clip_start:
            continue
        if float(segment["start"]) >= clip_end:
            break
        if segment["text"]:
            selected.append(str(segment["text"]))
        if len(selected) >= 2:
            break
    return " ".join(selected).strip()


def _end_snippet(transcript: Sequence[Dict[str, Any]], clip_start: float, clip_end: float) -> str:
    selected: List[str] = []
    for segment in reversed(list(transcript)):
        if float(segment["start"]) >= clip_end:
            continue
        if float(segment["end"]) <= clip_start:
            break
        if segment["text"]:
            selected.append(str(segment["text"]))
        if len(selected) >= 2:
            break
    return " ".join(reversed(selected)).strip()


def _merge_candidate_details(candidates: Sequence[Dict[str, Any]], tolerance_seconds: float) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        try:
            start = float(item.get("start"))
            end = float(item.get("end"))
            score = float(item.get("score", 0.0))
        except (TypeError, ValueError):
            continue
        normalized.append(
            {
                **item,
                "start": start,
                "end": end,
                "score": score,
            }
        )

    normalized.sort(key=lambda row: (row["start"], row["end"], -row["score"]))
    groups: List[Dict[str, Any]] = []
    for candidate in normalized:
        if not groups:
            groups.append(
                {
                    "start": candidate["start"],
                    "end": candidate["end"],
                    "score": candidate["score"],
                    "sources": [candidate],
                    "gaps_between_sources": [],
                }
            )
            continue

        previous = groups[-1]
        previous_end = float(previous["end"])
        if candidate["start"] <= previous_end + float(tolerance_seconds):
            gap = max(0.0, candidate["start"] - previous_end)
            previous["end"] = max(previous_end, candidate["end"])
            previous["score"] = max(float(previous["score"]), candidate["score"])
            previous["sources"].append(candidate)
            previous["gaps_between_sources"].append(gap)
        else:
            groups.append(
                {
                    "start": candidate["start"],
                    "end": candidate["end"],
                    "score": candidate["score"],
                    "sources": [candidate],
                    "gaps_between_sources": [],
                }
            )

    for index, group in enumerate(groups, start=1):
        source_durations = [max(0.0, float(item["end"]) - float(item["start"])) for item in group["sources"]]
        group["group_index"] = index
        group["duration"] = max(0.0, float(group["end"]) - float(group["start"]))
        group["max_gap_seconds"] = max(group["gaps_between_sources"] or [0.0])
        group["largest_source_duration"] = max(source_durations or [0.0])
    return groups


def _normalize_cluster_details(details: Any) -> Dict[Tuple[float, float], Dict[str, Any]]:
    normalized: Dict[Tuple[float, float], Dict[str, Any]] = {}
    if not isinstance(details, list):
        return normalized
    for item in details:
        if not isinstance(item, dict):
            continue
        try:
            start = float(item.get("start"))
            end = float(item.get("end"))
        except (TypeError, ValueError):
            continue
        key = (round(start, 3), round(end, 3))
        normalized[key] = dict(item)
    return normalized


class DiagnosticRecorder:
    def __init__(self, artifacts_dir: Optional[Path], logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger("verbacut.diagnostics")
        self.artifacts_dir = artifacts_dir.resolve() if artifacts_dir is not None else None
        self._state: Dict[str, Dict[str, Any]] = {}
        if self.artifacts_dir is not None:
            self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_config(
        cls,
        base_dir: Path,
        config: Dict[str, Any],
        logger: Optional[logging.Logger] = None,
    ) -> "DiagnosticRecorder":
        diagnostics_cfg = config.get("diagnostics", {})
        if not isinstance(diagnostics_cfg, dict) or not bool(diagnostics_cfg.get("enabled")):
            return cls(None, logger=logger)

        raw_dir = str(diagnostics_cfg.get("artifacts_dir", "system/diagnostics")).strip()
        artifacts_dir = Path(raw_dir)
        if not artifacts_dir.is_absolute():
            artifacts_dir = base_dir / artifacts_dir
        return cls(artifacts_dir=artifacts_dir, logger=logger)

    @property
    def enabled(self) -> bool:
        return self.artifacts_dir is not None

    def _video_dir(self, video_name: str) -> Optional[Path]:
        if self.artifacts_dir is None:
            return None
        directory = self.artifacts_dir / "videos" / _slugify(video_name)
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def _video_state(self, video_name: str) -> Dict[str, Any]:
        return self._state.setdefault(
            video_name,
            {
                "scan_attempts": [],
                "raw_candidates": [],
                "extracted_clips": [],
            },
        )

    def _write_video_json(self, video_name: str, filename: str, payload: Any) -> None:
        directory = self._video_dir(video_name)
        if directory is None:
            return
        save_json_file(directory / filename, payload)

    def record_run_metadata(self, payload: Dict[str, Any]) -> None:
        if self.artifacts_dir is None:
            return
        save_json_file(self.artifacts_dir / "run_metadata.json", payload)

    def record_run_summary(self, payload: Dict[str, Any]) -> None:
        if self.artifacts_dir is None:
            return
        save_json_file(self.artifacts_dir / "run_summary.json", payload)

    def record_video_metadata(self, video_name: str, payload: Dict[str, Any]) -> None:
        if self.artifacts_dir is None:
            return
        state = self._video_state(video_name)
        state["video_metadata"] = payload
        self._write_video_json(video_name, "video_metadata.json", payload)

    def record_transcript(
        self,
        video_name: str,
        transcript: Sequence[Sequence[Any]],
        *,
        used_checkpoint: bool,
    ) -> None:
        if self.artifacts_dir is None:
            return
        state = self._video_state(video_name)
        transcript_list = [list(item) for item in transcript]
        word_timestamp_segments = sum(
            1 for item in transcript_list if len(item) >= 4 and isinstance(item[3], list)
        )
        payload = {
            "video_name": video_name,
            "used_checkpoint": bool(used_checkpoint),
            "segment_count": len(transcript_list),
            "word_timestamp_segments": word_timestamp_segments,
            "transcript": transcript_list,
        }
        state["transcript"] = payload
        self._write_video_json(video_name, "transcript.json", payload)

    def record_chunking(
        self,
        video_name: str,
        *,
        base_chunks: Sequence[Sequence[Sequence[Any]]],
        scanned_chunks: Sequence[Sequence[Sequence[Any]]],
        bridge_count: int,
        meta: Dict[str, Any],
        used_checkpoint: bool,
    ) -> None:
        if self.artifacts_dir is None:
            return
        state = self._video_state(video_name)
        payload = {
            "video_name": video_name,
            "used_checkpoint": bool(used_checkpoint),
            "bridge_count": int(bridge_count),
            "base_chunk_count": len(base_chunks),
            "scanned_chunk_count": len(scanned_chunks),
            "meta": dict(meta),
            "base_chunks": [[list(segment) for segment in chunk] for chunk in base_chunks],
            "scanned_chunks": [[list(segment) for segment in chunk] for chunk in scanned_chunks],
        }
        state["chunking"] = payload
        self._write_video_json(video_name, "chunking.json", payload)

    def record_scan_attempt(self, video_name: str, payload: Dict[str, Any]) -> None:
        if self.artifacts_dir is None:
            return
        state = self._video_state(video_name)
        state["scan_attempts"].append(payload)
        self._write_video_json(
            video_name,
            "scan_attempts.json",
            {
                "video_name": video_name,
                "attempts": state["scan_attempts"],
            },
        )

    def record_raw_candidates(
        self,
        video_name: str,
        *,
        candidates: Sequence[Dict[str, Any]],
        raw_clip_groups: Sequence[Sequence[Sequence[float]]],
    ) -> None:
        if self.artifacts_dir is None:
            return
        state = self._video_state(video_name)
        normalized_candidates = [dict(item) for item in candidates]
        state["raw_candidates"] = normalized_candidates
        self._write_video_json(
            video_name,
            "raw_candidates.json",
            {
                "video_name": video_name,
                "candidates": normalized_candidates,
            },
        )
        self._write_video_json(
            video_name,
            "raw_clip_groups.json",
            {
                "video_name": video_name,
                "groups": [[list(row) for row in group] for group in raw_clip_groups],
            },
        )

    def record_merged_clips(
        self,
        video_name: str,
        *,
        merged_clips: Sequence[Sequence[float]],
        merge_distance_seconds: float,
        details: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        if self.artifacts_dir is None:
            return
        payload = {
            "video_name": video_name,
            "merge_distance_seconds": float(merge_distance_seconds),
            "clips": [list(row) for row in merged_clips],
            "details": [dict(item) for item in (details or [])],
        }
        self._video_state(video_name)["merged_clips"] = payload
        self._write_video_json(video_name, "merged_clips.json", payload)

    def record_final_clips(
        self,
        video_name: str,
        *,
        final_clips: Sequence[Sequence[float]],
        rejected_short: int,
        min_duration_seconds: float,
    ) -> None:
        if self.artifacts_dir is None:
            return
        payload = {
            "video_name": video_name,
            "rejected_short": int(rejected_short),
            "min_duration_seconds": float(min_duration_seconds),
            "clips": [list(row) for row in final_clips],
        }
        self._video_state(video_name)["final_clips"] = payload
        self._write_video_json(video_name, "final_clips.json", payload)

    def record_extracted_clip(
        self,
        video_name: str,
        *,
        clip_index: int,
        output_path: str,
        created: bool,
        requested_clip: Sequence[float],
    ) -> None:
        if self.artifacts_dir is None:
            return
        output = Path(output_path)
        actual_duration = _probe_duration_seconds(output)
        requested_duration = 0.0
        try:
            requested_duration = max(0.0, float(requested_clip[1]) - float(requested_clip[0]))
        except Exception:
            requested_duration = 0.0

        payload = {
            "clip_index": int(clip_index),
            "created": bool(created),
            "output_path": str(output),
            "exists": output.exists(),
            "requested_clip": [float(requested_clip[0]), float(requested_clip[1]), float(requested_clip[2])],
            "requested_duration": requested_duration,
            "actual_duration": actual_duration,
        }
        state = self._video_state(video_name)
        state["extracted_clips"] = [
            item for item in state.get("extracted_clips", []) if int(item.get("clip_index", -1)) != int(clip_index)
        ]
        state["extracted_clips"].append(payload)
        state["extracted_clips"].sort(key=lambda item: int(item.get("clip_index", 0)))
        self._write_video_json(
            video_name,
            "extracted_clips.json",
            {
                "video_name": video_name,
                "clips": state["extracted_clips"],
            },
        )


def generate_video_diagnostic_report(artifacts_dir: Path, video_name: str) -> Dict[str, Any]:
    video_dir = artifacts_dir / "videos" / _slugify(video_name)
    run_metadata = _load_optional_json(artifacts_dir / "run_metadata.json", {})
    transcript_payload = _load_optional_json(video_dir / "transcript.json", {})
    chunking_payload = _load_optional_json(video_dir / "chunking.json", {})
    attempts_payload = _load_optional_json(video_dir / "scan_attempts.json", {})
    raw_candidates_payload = _load_optional_json(video_dir / "raw_candidates.json", {})
    merged_payload = _load_optional_json(video_dir / "merged_clips.json", {})
    final_payload = _load_optional_json(video_dir / "final_clips.json", {})
    extraction_payload = _load_optional_json(video_dir / "extracted_clips.json", {})

    transcript = _coerce_transcript_segments(transcript_payload.get("transcript", []))
    attempts = attempts_payload.get("attempts", []) if isinstance(attempts_payload, dict) else []
    raw_candidates = raw_candidates_payload.get("candidates", []) if isinstance(raw_candidates_payload, dict) else []
    merged_clips = _coerce_clip_list(merged_payload.get("clips", []))
    merged_details = _normalize_cluster_details(merged_payload.get("details", []))
    final_clips = _coerce_clip_list(final_payload.get("clips", []))
    extracted_clips = extraction_payload.get("clips", []) if isinstance(extraction_payload, dict) else []
    merge_distance_seconds = float(merged_payload.get("merge_distance_seconds", 0.0))

    grouped_candidates = _merge_candidate_details(raw_candidates, tolerance_seconds=merge_distance_seconds)
    merged_lookup = {
        (round(float(group["start"]), 3), round(float(group["end"]), 3)): group for group in grouped_candidates
    }
    extracted_lookup = {
        int(item.get("clip_index", 0)): item for item in extracted_clips if isinstance(item, dict)
    }

    parse_failures = 0
    request_failures = 0
    empty_responses = 0
    malformed_responses = 0
    loop_count = 0
    for attempt in attempts:
        if not isinstance(attempt, dict):
            continue
        loop_count = max(loop_count, int(attempt.get("loop_index", 0) or 0))
        status = str(attempt.get("status", ""))
        raw_response = str(attempt.get("raw_response", "") or "")
        if status == "request_error":
            request_failures += 1
            continue
        if status == "parse_error":
            parse_failures += 1
            if not raw_response.strip():
                empty_responses += 1
            else:
                malformed_responses += 1

    chunk_count = int(chunking_payload.get("scanned_chunk_count", 0) or 0)
    word_timestamp_segments = int(transcript_payload.get("word_timestamp_segments", 0) or 0)

    clip_reports: List[Dict[str, Any]] = []
    issue_summary = {
        "merge_inflation": 0,
        "short_boundary_fragments": 0,
        "chunk_boundary_risk": 0,
        "extraction_timing": 0,
    }
    for index, clip in enumerate(final_clips, start=1):
        start = float(clip[0])
        end = float(clip[1])
        score = float(clip[2]) if len(clip) > 2 else 0.0
        merged_group = merged_lookup.get((round(start, 3), round(end, 3)))
        merged_detail = merged_details.get((round(start, 3), round(end, 3)), {})
        extracted = extracted_lookup.get(index, {})

        start_segment = _segment_for_start(transcript, start)
        end_segment = _segment_for_end(transcript, end)
        start_short = bool(
            start_segment is not None
            and (float(start_segment["duration"]) < 2.5 or int(start_segment["word_count"]) < 4)
        )
        end_short = bool(
            end_segment is not None
            and (float(end_segment["duration"]) < 2.5 or int(end_segment["word_count"]) < 4)
        )

        max_gap = float(merged_group.get("max_gap_seconds", 0.0)) if isinstance(merged_group, dict) else 0.0
        largest_source_duration = (
            float(merged_group.get("largest_source_duration", 0.0)) if isinstance(merged_group, dict) else 0.0
        )
        if isinstance(merged_detail, dict) and merged_detail:
            sources = merged_detail.get("sources", [])
            if isinstance(sources, list) and sources:
                source_ranges = []
                source_gaps = []
                sorted_sources = sorted(
                    [dict(item) for item in sources if isinstance(item, dict)],
                    key=lambda item: (float(item.get("start", 0.0)), float(item.get("end", 0.0))),
                )
                for source_index, source in enumerate(sorted_sources):
                    try:
                        source_start = float(source.get("start"))
                        source_end = float(source.get("end"))
                    except (TypeError, ValueError):
                        continue
                    source_ranges.append(max(0.0, source_end - source_start))
                    if source_index > 0:
                        previous_end = float(sorted_sources[source_index - 1].get("end", source_start))
                        source_gaps.append(max(0.0, source_start - previous_end))
                max_gap = max(source_gaps or [0.0])
                largest_source_duration = max(source_ranges or [0.0])
        clip_duration = max(0.0, end - start)
        ratio = clip_duration / max(0.001, largest_source_duration) if largest_source_duration > 0 else 0.0
        issue_flags: List[str] = []

        if max_gap >= 5.0:
            issue_flags.append("merge_inflation_gap")
            issue_summary["merge_inflation"] += 1
        elif ratio >= 1.75 and isinstance(merged_group, dict) and len(merged_group.get("sources", [])) > 1:
            issue_flags.append("merge_inflation_ratio")
            issue_summary["merge_inflation"] += 1

        if start_short or end_short:
            issue_flags.append("short_boundary_fragment")
            issue_summary["short_boundary_fragments"] += 1

        touches_chunk_edges = bool(
            isinstance(merged_group, dict)
            and any(
                bool(source.get("touches_chunk_start_edge")) or bool(source.get("touches_chunk_end_edge"))
                for source in merged_group.get("sources", [])
                if isinstance(source, dict)
            )
        )
        if touches_chunk_edges:
            issue_flags.append("chunk_boundary_risk")
            issue_summary["chunk_boundary_risk"] += 1

        actual_duration = extracted.get("actual_duration")
        if actual_duration is not None:
            duration_delta = abs(float(actual_duration) - clip_duration)
            if duration_delta > _duration_tolerance_seconds(clip_duration):
                issue_flags.append("extraction_timing_mismatch")
                issue_summary["extraction_timing"] += 1
        else:
            duration_delta = None

        sources = []
        if isinstance(merged_detail, dict) and isinstance(merged_detail.get("sources"), list):
            sources = list(merged_detail.get("sources", []))
        elif isinstance(merged_group, dict):
            sources = list(merged_group.get("sources", []))
        clip_reports.append(
            {
                "clip_index": index,
                "start": start,
                "end": end,
                "score": score,
                "duration": clip_duration,
                "issue_flags": issue_flags,
                "source_chunk_ids": sorted(
                    {
                        int(source.get("chunk_index"))
                        for source in sources
                        if isinstance(source, dict) and str(source.get("chunk_index", "")).strip()
                    }
                ),
                "source_loop_ids": sorted(
                    {
                        int(source.get("loop_index"))
                        for source in sources
                        if isinstance(source, dict) and str(source.get("loop_index", "")).strip()
                    }
                ),
                "source_candidate_count": len(sources),
                "gaps_between_sources": list(merged_group.get("gaps_between_sources", []))
                if isinstance(merged_group, dict)
                else [],
                "max_gap_seconds": max_gap,
                "largest_source_duration": largest_source_duration,
                "merged_to_largest_source_ratio": ratio if largest_source_duration > 0 else None,
                "touches_chunk_edges": touches_chunk_edges,
                "start_boundary": start_segment,
                "end_boundary": end_segment,
                "start_snippet": _start_snippet(transcript, start, end),
                "end_snippet": _end_snippet(transcript, start, end),
                "extraction": extracted,
                "extraction_duration_delta": duration_delta,
            }
        )

    preservation = {
        "original_video_path": str(run_metadata.get("original_video_path", "") or ""),
        "original_video_preserved": bool(run_metadata.get("original_video_preserved", False)),
        "scratch_input_copy_path": str(run_metadata.get("scratch_input_copy_path", "") or ""),
        "scratch_input_exists_after_run": bool(run_metadata.get("scratch_input_exists_after_run", False)),
        "archived_scratch_copies": list(run_metadata.get("archived_scratch_copies", []) or []),
    }

    report = {
        "video_name": video_name,
        "overview": {
            "transcript_segments": len(transcript),
            "word_timestamp_segments": word_timestamp_segments,
            "word_timestamps_present": word_timestamp_segments > 0,
            "base_chunk_count": int(chunking_payload.get("base_chunk_count", 0) or 0),
            "scanned_chunk_count": chunk_count,
            "loop_count": loop_count,
            "scan_attempt_count": len(attempts),
            "request_failures": request_failures,
            "parse_failures": parse_failures,
            "empty_responses": empty_responses,
            "malformed_responses": malformed_responses,
            "raw_candidate_count": len(raw_candidates),
            "merged_candidate_count": len(merged_clips),
            "final_clip_count": len(final_clips),
            "extracted_clip_count": len(extracted_clips),
        },
        "preservation_checks": preservation,
        "issue_summary": issue_summary,
        "clip_reports": clip_reports,
    }

    save_json_file(video_dir / "report.json", report)
    markdown_lines = [
        f"# Diagnostic Report: {video_name}",
        "",
        "## Overview",
        f"- Transcript segments: {report['overview']['transcript_segments']}",
        f"- Word timestamp segments: {report['overview']['word_timestamp_segments']}",
        f"- Base chunks: {report['overview']['base_chunk_count']}",
        f"- Scanned chunks: {report['overview']['scanned_chunk_count']}",
        f"- Loops: {report['overview']['loop_count']}",
        f"- Scan attempts: {report['overview']['scan_attempt_count']}",
        f"- Request failures: {report['overview']['request_failures']}",
        f"- Parse failures: {report['overview']['parse_failures']}",
        f"- Empty responses: {report['overview']['empty_responses']}",
        f"- Malformed responses: {report['overview']['malformed_responses']}",
        f"- Raw candidates: {report['overview']['raw_candidate_count']}",
        f"- Merged candidates: {report['overview']['merged_candidate_count']}",
        f"- Final clips: {report['overview']['final_clip_count']}",
        "",
        "## Preservation Checks",
        f"- Original preserved: {preservation['original_video_preserved']}",
        f"- Scratch input still in input/: {preservation['scratch_input_exists_after_run']}",
        f"- Archived scratch copies: {len(preservation['archived_scratch_copies'])}",
        "",
        "## Clip Findings",
    ]
    if not clip_reports:
        markdown_lines.append("- No final clips were produced.")
    else:
        for clip_report in clip_reports:
            flags = ", ".join(clip_report["issue_flags"]) if clip_report["issue_flags"] else "none"
            markdown_lines.extend(
                [
                    (
                        f"- Clip {clip_report['clip_index']}: "
                        f"{clip_report['start']:.2f}s -> {clip_report['end']:.2f}s "
                        f"(flags: {flags})"
                    ),
                    f"  start snippet: {clip_report['start_snippet'] or '<empty>'}",
                    f"  end snippet: {clip_report['end_snippet'] or '<empty>'}",
                ]
            )
    (video_dir / "report.md").write_text("\n".join(markdown_lines), encoding="utf-8")
    return report
