from __future__ import annotations

import json
import logging
import math
import re
from typing import Any, Callable, List, Optional, Sequence

import requests
from utils.diagnostics import DiagnosticRecorder


try:
    import tiktoken

    _TOKENIZER = tiktoken.get_encoding("cl100k_base")
except Exception:
    _TOKENIZER = None


HARD_JSON_GUARDRAIL = (
    "NON-NEGOTIABLE OUTPUT CONTRACT:\n"
    "- Reply with one valid JSON array only.\n"
    "- No markdown fences, no prose, no notes.\n"
    "- Each item must be exactly [start, end, score].\n"
    "- start/end/score must be numeric.\n"
    "- If no valid clips exist, return [].\n"
    "- Never return a blank response, {}, null, or any JSON object."
)

RELEVANCE_GUARDRAIL = (
    "RELEVANCE CONTRACT:\n"
    "- Pick clips with meaningful spoken sentence-level content that matches the user query.\n"
    "- Reject clips that are mostly music, intro/outro filler, grunts, or non-verbal effort sounds.\n"
    "- Respect explicit duration constraints written in the user query.\n"
    "- Prefer coherent complete thoughts with natural start/end boundaries.\n"
    "- For 'at least X seconds', treat X as a minimum floor, not an exact target."
)

SPAN_GUARDRAIL = (
    "SPAN CONTRACT:\n"
    "- Every returned clip must already be one continuous self-contained passage.\n"
    "- The whole returned span must match the user query, not just one sentence inside it.\n"
    "- Prefer one strong complete passage over several tiny highlight fragments.\n"
    "- Do not return narrow fragments that would need downstream merging to make sense.\n"
    "- Prefer start/end points that align to transcript segment boundaries and natural pauses."
)

NUMERIC_TRIPLE_PATTERN = re.compile(
    r"\[\s*[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?\s*,\s*"
    r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?\s*,\s*"
    r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?\s*\]"
)

NO_MATCH_OBJECT_KEYS = ("clips", "matches", "candidates", "results", "items", "output")


def estimate_tokens(text: str) -> int:
    text = str(text)
    if _TOKENIZER is not None:
        try:
            return len(_TOKENIZER.encode(text))
        except Exception:
            pass
    return max(1, math.floor(len(text) / 3.5))


def _segment_tokens(start: float, end: float, text: str) -> int:
    return estimate_tokens(f"{start} {end} {text}")


def _format_timestamp(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    return f"{numeric:.3f}"


def _serialize_chunk_for_model(chunk: Sequence[Sequence[Any]]) -> str:
    lines: List[str] = []
    for index, segment in enumerate(chunk, start=1):
        if len(segment) < 3:
            continue
        text = " ".join(str(segment[2]).split())
        lines.append(
            f"{index:03d} | {_format_timestamp(segment[0])} | {_format_timestamp(segment[1])} | {text}"
        )
    return "\n".join(lines)


def _normalize_timed_words(segment: Sequence[Any]) -> List[List[Any]]:
    if len(segment) < 4 or not isinstance(segment[3], list):
        return []

    normalized: List[List[Any]] = []
    for word in segment[3]:
        if not isinstance(word, (list, tuple)) or len(word) < 3:
            continue
        try:
            start = float(word[0])
            end = float(word[1])
        except (TypeError, ValueError):
            continue
        text = str(word[2])
        if not text:
            continue
        if end < start:
            end = start
        normalized.append([start, end, text])
    return normalized


def _words_to_segment(words: Sequence[Sequence[Any]]) -> List[Any]:
    text = "".join(str(word[2]) for word in words).strip()
    if not text:
        text = " ".join(str(word[2]).strip() for word in words if str(word[2]).strip())
    start = float(words[0][0])
    end = float(words[-1][1])
    if end < start:
        end = start
    return [start, end, text]


def _split_segment_with_timed_words(segment: Sequence[Any], budget: int) -> List[List[Any]]:
    words = _normalize_timed_words(segment)
    if not words:
        return []

    windows: List[List[Any]] = []
    current_words: List[List[Any]] = []
    for word in words:
        candidate_words = current_words + [word]
        candidate_segment = _words_to_segment(candidate_words)
        candidate_tokens = _segment_tokens(
            float(candidate_segment[0]),
            float(candidate_segment[1]),
            str(candidate_segment[2]),
        )
        if candidate_tokens > budget and current_words:
            windows.append(_words_to_segment(current_words))
            current_words = [word]
            continue
        current_words = candidate_words

    if current_words:
        windows.append(_words_to_segment(current_words))
    return windows


def _split_segment_with_estimated_ranges(
    start: float,
    end: float,
    text: str,
    budget: int,
) -> List[List[Any]]:
    words = text.split()
    if not words:
        return []

    total_words = len(words)
    duration = max(0.0, end - start)
    windows: List[List[Any]] = []
    current_words: List[str] = []
    current_start_index = 0

    def flush_window(start_index: int, window_words: List[str]) -> None:
        if not window_words:
            return
        word_count = len(window_words)
        window_start = start + (duration * (start_index / total_words))
        window_end = start + (duration * ((start_index + word_count) / total_words))
        if window_end < window_start:
            window_end = window_start
        windows.append([window_start, window_end, " ".join(window_words)])

    for index, word in enumerate(words):
        candidate_words = current_words + [word]
        candidate_text = " ".join(candidate_words)
        candidate_start = start + (duration * (current_start_index / total_words))
        candidate_end = start + (duration * ((current_start_index + len(candidate_words)) / total_words))
        candidate_tokens = _segment_tokens(candidate_start, candidate_end, candidate_text)
        if candidate_tokens > budget and current_words:
            flush_window(current_start_index, current_words)
            current_words = [word]
            current_start_index = index
            continue

        if not current_words:
            current_start_index = index
        current_words = candidate_words

    flush_window(current_start_index, current_words)
    return windows


def chunk_transcript(
    transcript: Sequence[Sequence[Any]],
    max_tokens: int,
    safety_margin: int = 300,
    overlap_segments: int = 0,
) -> List[List[List[Any]]]:
    """
    Split transcript into chunk lists while respecting a max token budget.
    Transcript format: [[start, end, text], ...]
    """
    budget = max(256, int(max_tokens) - max(0, int(safety_margin)))
    chunks: List[List[List[Any]]] = []
    current_chunk: List[List[Any]] = []
    current_tokens = 0

    for segment in transcript:
        if len(segment) < 3:
            continue
        start, end, text = segment[0], segment[1], str(segment[2])
        serialized = f"{start} {end} {text}"
        seg_tokens = estimate_tokens(serialized)

        if seg_tokens > budget:
            mini_segments = _split_segment_with_timed_words(segment, budget)
            if not mini_segments:
                mini_segments = _split_segment_with_estimated_ranges(start, end, text, budget)
            if not mini_segments:
                continue
            for mini_start, mini_end, mini_text in mini_segments:
                mini_tokens = _segment_tokens(float(mini_start), float(mini_end), str(mini_text))
                if current_tokens + mini_tokens > budget and current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = []
                    current_tokens = 0
                current_chunk.append([float(mini_start), float(mini_end), str(mini_text)])
                current_tokens += mini_tokens
            continue

        if current_tokens + seg_tokens > budget and current_chunk:
            chunks.append(current_chunk)
            current_chunk = []
            current_tokens = 0

        current_chunk.append([start, end, text])
        current_tokens += seg_tokens

    if current_chunk:
        chunks.append(current_chunk)

    overlap = max(0, int(overlap_segments))
    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    overlapped_chunks: List[List[List[Any]]] = [chunks[0]]
    for index in range(1, len(chunks)):
        previous = chunks[index - 1]
        current = chunks[index]
        prefix = [list(item) for item in previous[-overlap:]]
        merged = prefix + [list(item) for item in current]
        while prefix and estimate_tokens(json.dumps(merged, ensure_ascii=True)) > budget:
            prefix = prefix[1:]
            merged = prefix + [list(item) for item in current]
        overlapped_chunks.append(merged)

    return overlapped_chunks


def _strip_model_output(raw_text: str) -> str:
    cleaned = str(raw_text or "").strip()
    if "</think>" in cleaned:
        cleaned = cleaned.rsplit("</think>", maxsplit=1)[-1].strip()

    fence_match = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.IGNORECASE | re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    if cleaned.lower().startswith("json"):
        cleaned = cleaned[4:].strip()
    return cleaned


def _clean_model_output(raw_text: str) -> str:
    cleaned = _strip_model_output(raw_text)
    if not cleaned:
        raise ValueError("Model returned empty response.")
    return cleaned


def _normalize_no_match_response(raw_text: str) -> tuple[str, str]:
    cleaned = _strip_model_output(raw_text)
    if not cleaned:
        return "[]", "empty_response"

    try:
        payload = json.loads(cleaned)
    except Exception:
        lowered = cleaned.lower()
        if lowered in {"{}", "null", "none"}:
            return "[]", "explicit_no_match_token"
        return cleaned, ""

    if payload is None:
        return "[]", "null_payload"

    if isinstance(payload, dict):
        if not payload:
            return "[]", "empty_object"

        for key in NO_MATCH_OBJECT_KEYS:
            if key not in payload:
                continue
            value = payload.get(key)
            if value is None or value == []:
                return "[]", f"{key}_empty"
            if isinstance(value, list):
                return json.dumps(value, ensure_ascii=True), f"{key}_unwrapped"

        text_values = " ".join(str(value).strip().lower() for value in payload.values() if value is not None)
        if any(token in text_values for token in ("no clip", "no clips", "no match", "nothing found", "none found")):
            return "[]", "no_match_message"

    return cleaned, ""


def _extract_json_text(raw_text: str) -> str:
    cleaned = _clean_model_output(raw_text)

    try:
        json.loads(cleaned)
        return cleaned
    except Exception:
        pass

    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON array found in model output.")
    return cleaned[start : end + 1]


def _validate_clip_schema(payload: Any) -> List[List[float]]:
    if not isinstance(payload, list):
        raise ValueError("Model output must be a JSON list.")

    validated: List[List[float]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, (list, tuple)):
            raise ValueError(f"Item #{index} is not a list.")
        if len(item) != 3:
            raise ValueError(f"Item #{index} must match [start, end, score].")

        try:
            start = float(item[0])
            end = float(item[1])
            score = float(item[2])
        except Exception as exc:
            raise ValueError(f"Item #{index} has non-numeric values: {exc}") from exc

        if end <= start:
            raise ValueError(f"Item #{index} has invalid timing: end <= start.")

        normalized_score = max(1.0, min(10.0, score))
        validated.append([start, end, normalized_score])

    deduped: List[List[float]] = []
    seen: set[tuple[float, float, float]] = set()
    for start, end, score in sorted(validated, key=lambda row: (row[0], row[1], -row[2])):
        key = (round(start, 3), round(end, 3), round(score, 2))
        if key in seen:
            continue
        seen.add(key)
        deduped.append([start, end, score])

    return deduped


def _parse_from_lines(cleaned: str) -> List[List[float]]:
    parsed_items: List[List[float]] = []
    for line in cleaned.splitlines():
        candidate = line.strip().rstrip(",")
        if not candidate:
            continue
        try:
            line_payload = json.loads(candidate)
        except Exception:
            continue

        if isinstance(line_payload, list) and line_payload and isinstance(line_payload[0], (int, float)):
            parsed_items.append(line_payload)
            continue
        if isinstance(line_payload, list):
            for item in line_payload:
                if isinstance(item, list):
                    parsed_items.append(item)

    if not parsed_items:
        return []
    return _validate_clip_schema(parsed_items)


def _parse_from_numeric_triples(cleaned: str) -> List[List[float]]:
    matches = NUMERIC_TRIPLE_PATTERN.findall(cleaned)
    if not matches:
        return []
    triples: List[List[float]] = []
    for match in matches:
        try:
            payload = json.loads(match)
        except Exception:
            continue
        if isinstance(payload, list):
            triples.append(payload)
    if not triples:
        return []
    return _validate_clip_schema(triples)


def parse_clip_response(raw_text: str) -> List[List[float]]:
    normalized_raw_text, _ = _normalize_no_match_response(raw_text)
    cleaned = _clean_model_output(normalized_raw_text)
    parse_errors: List[str] = []

    candidates: List[str] = [cleaned]
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start != -1 and end != -1 and end > start:
        candidates.append(cleaned[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            return _validate_clip_schema(parsed)
        except Exception as exc:
            parse_errors.append(str(exc))

    parsed_from_lines = _parse_from_lines(cleaned)
    parsed_from_triples = _parse_from_numeric_triples(cleaned)
    if parsed_from_lines and parsed_from_triples:
        merged = parsed_from_lines + parsed_from_triples
        return _validate_clip_schema(merged)
    if parsed_from_lines:
        return parsed_from_lines
    if parsed_from_triples:
        return parsed_from_triples

    details = "; ".join(parse_errors[-2:]) if parse_errors else "No valid JSON structure detected."
    raise ValueError(details)


class AIPipeline:
    """Ollama chat + strict JSON parsing with retry/repair logic."""

    def __init__(
        self,
        model: str,
        ollama_url: str,
        system_prompt: str,
        temperature: float,
        max_output_tokens: int,
        max_context_tokens: int,
        logger: logging.Logger,
        diagnostics: Optional[DiagnosticRecorder] = None,
    ) -> None:
        self.model = model
        self.ollama_url = ollama_url.rstrip("/")
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.max_context_tokens = max_context_tokens
        self.logger = logger
        self.diagnostics = diagnostics

    def _resolve_think_setting(self) -> Any:
        """
        Request lowest-thinking mode for latency across models.
        If a model/runtime does not support `think`, `_chat` retries without it.
        """
        return "low"

    def _build_chat_payload(
        self,
        prompt: str,
        system_message: str,
        force_json: bool = False,
        num_predict_override: Optional[int] = None,
    ) -> dict[str, Any]:
        temperature = self.temperature if not force_json else min(self.temperature, 0.15)
        predict_tokens = int(num_predict_override or self.max_output_tokens)
        predict_tokens = max(32, predict_tokens)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "think": self._resolve_think_setting(),
            "options": {
                "temperature": temperature,
                "num_predict": predict_tokens,
                "num_ctx": self.max_context_tokens,
            },
        }
        if force_json:
            payload["format"] = "json"
        return payload

    def _chat_payload(self, payload: dict[str, Any]) -> str:
        endpoint = f"{self.ollama_url}/api/chat"
        response = requests.post(
            endpoint,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=180,
        )
        if response.status_code >= 400 and "think" in payload:
            self.logger.debug("Model rejected explicit think setting; retrying chat request without `think`.")
            fallback_payload = dict(payload)
            fallback_payload.pop("think", None)
            response = requests.post(
                endpoint,
                headers={"Content-Type": "application/json"},
                data=json.dumps(fallback_payload),
                timeout=180,
            )
        response.raise_for_status()
        body = response.json()
        message = body.get("message", {})
        content = str(message.get("content", "")).strip()
        if not content:
            content = str(body.get("response", "")).strip()
        return content

    def _build_scan_request(
        self,
        chunk: Sequence[Sequence[Any]],
        user_query: str,
        strict_mode: bool = False,
    ) -> tuple[str, str, dict[str, Any]]:
        chunking_context = (
            "Chunking context:\n"
            "- Treat the supplied chunk as the maximum available context window for this decision.\n"
            "- Read broadly across that window before choosing any clip; a strong quote is not enough if its setup or resolution changes its meaning.\n"
            "- Select a span only when the available context confirms its premise, development, and conclusion.\n"
            "- Avoid leading or trailing chunk edges if the available context suggests the thought is incomplete.\n"
            "- Output only JSON."
        )
        serialized_chunk = _serialize_chunk_for_model(chunk)
        chunk_start = _format_timestamp(chunk[0][0]) if chunk else "0.000"
        chunk_end = _format_timestamp(chunk[-1][1]) if chunk else "0.000"
        prompt = (
            "Transcript chunk metadata:\n"
            f"- Chunk start: {chunk_start}\n"
            f"- Chunk end: {chunk_end}\n"
            f"- Segment count: {len(chunk)}\n"
            "- Transcript lines use the format: segment_id | start | end | text\n\n"
            f"Transcript chunk:\n{serialized_chunk}\n\n"
            f"User query:\n{user_query}\n\n"
            "Selection rules:\n"
            "- Use the widest relevant context in this chunk to evaluate every candidate before scoring it.\n"
            "- Return spans that work as complete clips on their own.\n"
            "- Prefer broader continuous passages over isolated standout lines.\n"
            "- Avoid returning several tiny fragments from the same thought.\n"
            "- Use the provided timestamps as the clip boundaries you return.\n"
            "- If nothing in this chunk qualifies, return [].\n\n"
            "Return only JSON list of clips: [[start, end, score], ...]."
        )
        system_message = (
            f"{self.system_prompt}\n\n{chunking_context}\n\n{RELEVANCE_GUARDRAIL}\n\n{SPAN_GUARDRAIL}"
        )
        if strict_mode:
            system_message = f"{system_message}\n\n{HARD_JSON_GUARDRAIL}"
        payload = self._build_chat_payload(
            prompt=prompt,
            system_message=system_message,
            force_json=strict_mode,
            num_predict_override=min(self.max_output_tokens, 220),
        )
        return prompt, system_message, payload

    def _record_scan_attempt(
        self,
        chunk: Sequence[Sequence[Any]],
        *,
        prompt: str,
        system_message: str,
        request_payload: dict[str, Any],
        diagnostic_context: Optional[dict[str, Any]],
        attempt_index: int,
        strict_mode: bool,
        status: str,
        raw_response: str,
        parsed_output: Optional[Sequence[Sequence[float]]] = None,
        error: str = "",
        response_normalization: str = "",
    ) -> None:
        if self.diagnostics is None or not self.diagnostics.enabled or not diagnostic_context:
            return
        video_name = str(diagnostic_context.get("video_name", "")).strip()
        if not video_name:
            return
        payload = {
            "chunk_index": int(diagnostic_context.get("chunk_index", 0) or 0),
            "loop_index": int(diagnostic_context.get("loop_index", 0) or 0),
            "attempt_index": int(attempt_index),
            "strict_mode": bool(strict_mode),
            "status": status,
            "error": str(error or ""),
            "raw_response": str(raw_response or ""),
            "parsed_output": [list(row) for row in (parsed_output or [])],
            "chunk": [list(row) for row in chunk],
            "prompt": prompt,
            "system_message": system_message,
            "request_payload": request_payload,
            "response_normalization": str(response_normalization or ""),
        }
        self.diagnostics.record_scan_attempt(video_name=video_name, payload=payload)

    def scan_chunk_with_retries(
        self,
        chunk: Sequence[Sequence[Any]],
        user_query: str,
        max_retries: int = 3,
        diagnostic_context: Optional[dict[str, Any]] = None,
    ) -> List[List[float]]:
        prompt_hint = ""
        for attempt in range(1, max_retries + 1):
            strict_mode = attempt > 1
            prompt, system_message, request_payload = self._build_scan_request(
                chunk=chunk,
                user_query=f"{user_query}\n{prompt_hint}".strip(),
                strict_mode=strict_mode,
            )
            try:
                raw = self._chat_payload(request_payload)
            except Exception as exc:
                self.logger.warning("Ollama request failed on attempt %s: %s", attempt, exc)
                self._record_scan_attempt(
                    chunk,
                    prompt=prompt,
                    system_message=system_message,
                    request_payload=request_payload,
                    diagnostic_context=diagnostic_context,
                    attempt_index=attempt,
                    strict_mode=strict_mode,
                    status="request_error",
                    raw_response="",
                    error=str(exc),
                )
                prompt_hint = (
                    "Previous request failed. Follow the non-negotiable JSON output contract exactly."
                )
                continue

            self.logger.debug(
                "Raw model output (attempt %s, %s chars): %s",
                attempt,
                len(raw),
                raw[:2500],
            )
            normalized_raw, normalization_reason = _normalize_no_match_response(raw)
            if normalization_reason:
                self.logger.debug(
                    "Normalized model response to []/array on attempt %s via %s.",
                    attempt,
                    normalization_reason,
                )

            try:
                parsed = parse_clip_response(normalized_raw)
                self._record_scan_attempt(
                    chunk,
                    prompt=prompt,
                    system_message=system_message,
                    request_payload=request_payload,
                    diagnostic_context=diagnostic_context,
                    attempt_index=attempt,
                    strict_mode=strict_mode,
                    status="success",
                    raw_response=raw,
                    parsed_output=parsed,
                    response_normalization=normalization_reason,
                )
                return parsed
            except Exception as exc:
                self.logger.warning("Failed to parse AI response on attempt %s: %s", attempt, exc)
                self._record_scan_attempt(
                    chunk,
                    prompt=prompt,
                    system_message=system_message,
                    request_payload=request_payload,
                    diagnostic_context=diagnostic_context,
                    attempt_index=attempt,
                    strict_mode=strict_mode,
                    status="parse_error",
                    raw_response=raw,
                    error=str(exc),
                    response_normalization=normalization_reason,
                )
                prompt_hint = (
                    "Your previous response violated the format. "
                    "Return exactly one valid JSON array with schema: [[start, end, score], ...]. "
                    "If no clip qualifies, return [] exactly."
                )
                continue

        self.logger.error("Failed to parse valid AI clip JSON after %s attempts.", max_retries)
        return []

    def scan_all_chunks(
        self,
        chunks: Sequence[Sequence[Sequence[Any]]],
        user_query: str,
        ai_loops: int,
        progress_callback: Optional[Callable[[int, int, int, int, int], None]] = None,
    ) -> List[List[List[float]]]:
        all_outputs: List[List[List[float]]] = []
        loop_count = max(1, int(ai_loops))
        total_chunks = len(chunks)
        for chunk_index, chunk in enumerate(chunks, start=1):
            combined: List[List[float]] = []
            for loop_index in range(1, loop_count + 1):
                output = self.scan_chunk_with_retries(chunk=chunk, user_query=user_query)
                if output:
                    combined.extend(output)
                if progress_callback:
                    progress_callback(
                        chunk_index,
                        total_chunks,
                        loop_index,
                        loop_count,
                        len(output),
                    )
            all_outputs.append(combined)
        return all_outputs
