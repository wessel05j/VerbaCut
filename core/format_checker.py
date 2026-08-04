from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


QUALITY_2160 = 3
QUALITY_1440 = 2
QUALITY_1080 = 1
QUALITY_BELOW_1080 = 0

QUALITY_LABELS = {
    QUALITY_2160: "2160-class",
    QUALITY_1440: "1440-class",
    QUALITY_1080: "1080-class",
    QUALITY_BELOW_1080: "below-1080",
}


@dataclass
class FormatDecision:
    format_string: Optional[str]
    strategy: str
    acceptable: bool
    quality_rank: int
    quality_label: str
    format_id: str = ""
    width: int = 0
    height: int = 0
    vcodec: str = ""
    acodec: str = ""
    bitrate: float = 0.0
    source_type: str = ""
    reason: str = ""


def _to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _to_float(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except Exception:
        return 0.0


def _is_video_only(fmt: Dict[str, Any]) -> bool:
    return fmt.get("vcodec") not in (None, "none") and fmt.get("acodec") in (None, "none")


def _is_progressive(fmt: Dict[str, Any]) -> bool:
    return fmt.get("vcodec") not in (None, "none") and fmt.get("acodec") not in (None, "none")


def classify_quality(width: Optional[int], height: Optional[int], format_note: str = "") -> tuple[int, str]:
    width_value = int(width or 0)
    height_value = int(height or 0)
    note = str(format_note or "").lower()

    if width_value >= 3840 or "2160" in note or height_value >= 1600:
        return QUALITY_2160, QUALITY_LABELS[QUALITY_2160]
    if width_value >= 2560 or "1440" in note or height_value >= 1200:
        return QUALITY_1440, QUALITY_LABELS[QUALITY_1440]
    if width_value >= 1920 or "1080" in note or height_value >= 800:
        return QUALITY_1080, QUALITY_LABELS[QUALITY_1080]
    return QUALITY_BELOW_1080, QUALITY_LABELS[QUALITY_BELOW_1080]


def _candidate_from_format(fmt: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(fmt, dict):
        return None
    format_id = str(fmt.get("format_id") or "").strip()
    if not format_id:
        return None
    if not (_is_video_only(fmt) or _is_progressive(fmt)):
        return None

    width = _to_int(fmt.get("width")) or 0
    height = _to_int(fmt.get("height")) or 0
    bitrate = _to_float(fmt.get("tbr") or fmt.get("vbr"))
    quality_rank, quality_label = classify_quality(width, height, str(fmt.get("format_note") or ""))
    source_type = "video-only" if _is_video_only(fmt) else "progressive"

    return {
        "format_id": format_id,
        "width": width,
        "height": height,
        "bitrate": bitrate,
        "quality_rank": quality_rank,
        "quality_label": quality_label,
        "source_type": source_type,
        "vcodec": str(fmt.get("vcodec") or ""),
        "acodec": str(fmt.get("acodec") or ""),
    }


def _candidate_sort_key(candidate: Dict[str, Any]) -> tuple[int, int, float, int, int]:
    source_preference = 1 if candidate.get("source_type") == "video-only" else 0
    return (
        int(candidate.get("quality_rank", QUALITY_BELOW_1080)),
        source_preference,
        float(candidate.get("bitrate", 0.0)),
        int(candidate.get("width", 0)),
        int(candidate.get("height", 0)),
    )


def _decision_from_candidate(candidate: Dict[str, Any], *, acceptable: bool, reason: str = "") -> FormatDecision:
    source_type = str(candidate.get("source_type") or "")
    format_id = str(candidate.get("format_id") or "")
    quality_label = str(candidate.get("quality_label") or QUALITY_LABELS[QUALITY_BELOW_1080])
    if acceptable:
        if source_type == "video-only":
            format_string = f"{format_id}+bestaudio[acodec!=none]/{format_id}+bestaudio/{format_id}"
            strategy = f"{quality_label}-video-plus-audio"
        else:
            format_string = format_id
            strategy = f"{quality_label}-progressive"
    else:
        format_string = None
        strategy = f"{quality_label}-rejected"

    return FormatDecision(
        format_string=format_string,
        strategy=strategy,
        acceptable=acceptable,
        quality_rank=int(candidate.get("quality_rank", QUALITY_BELOW_1080)),
        quality_label=quality_label,
        format_id=format_id,
        width=int(candidate.get("width", 0)),
        height=int(candidate.get("height", 0)),
        vcodec=str(candidate.get("vcodec") or ""),
        acodec=str(candidate.get("acodec") or ""),
        bitrate=float(candidate.get("bitrate", 0.0)),
        source_type=source_type,
        reason=reason,
    )


def choose_download_format(
    extracted_info: Dict[str, Any],
    maximum_quality_rank: Optional[int] = None,
) -> FormatDecision:
    """
    Select the highest acceptable stream across the full format inventory.
    Preference order:
    1) Highest available quality tier (2160-class > 1440-class > 1080-class)
    2) Within the same tier, prefer video-only + bestaudio over progressive
    3) Within the same tier and source type, prefer higher bitrate
    """
    formats = extracted_info.get("formats") or []
    if not isinstance(formats, list) or not formats:
        return FormatDecision(
            format_string=None,
            strategy="no-format-data",
            acceptable=False,
            quality_rank=QUALITY_BELOW_1080,
            quality_label=QUALITY_LABELS[QUALITY_BELOW_1080],
            reason="No downloadable formats were exposed by yt-dlp.",
        )

    candidates = [candidate for fmt in formats if (candidate := _candidate_from_format(fmt)) is not None]
    if not candidates:
        return FormatDecision(
            format_string=None,
            strategy="no-video-candidates",
            acceptable=False,
            quality_rank=QUALITY_BELOW_1080,
            quality_label=QUALITY_LABELS[QUALITY_BELOW_1080],
            reason="No valid video streams were exposed by yt-dlp.",
        )

    if maximum_quality_rank is not None:
        bounded = [
            candidate
            for candidate in candidates
            if int(candidate.get("quality_rank", QUALITY_BELOW_1080)) <= int(maximum_quality_rank)
        ]
        if bounded:
            candidates = bounded

    candidates.sort(key=_candidate_sort_key, reverse=True)
    best = candidates[0]
    quality_rank = int(best.get("quality_rank", QUALITY_BELOW_1080))
    if quality_rank < QUALITY_1080:
        return _decision_from_candidate(
            best,
            acceptable=False,
            reason="Best available stream is below the 1080-class minimum quality floor.",
        )

    return _decision_from_candidate(best, acceptable=True)
