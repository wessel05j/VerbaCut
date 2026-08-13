from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import parse_qs, urlparse

import requests
import yt_dlp

from core.format_checker import QUALITY_1080, FormatDecision, choose_download_format, classify_quality


class _SilentYDLLogger:
    def debug(self, _: str) -> None:
        return None

    def warning(self, _: str) -> None:
        return None

    def error(self, _: str) -> None:
        return None


@dataclass
class DownloadResult:
    status: str
    path: Optional[str] = None
    strategy: Optional[str] = None
    error: Optional[str] = None


@dataclass
class FormatProbeResult:
    acceptable: bool
    strategy: Optional[str] = None
    decision: Optional[FormatDecision] = None
    error: Optional[str] = None


class YTHandler:
    DEFAULT_BROWSERS = ("firefox", "edge", "chrome", "brave", "opera", "vivaldi")
    VIDEO_FILE_EXTENSIONS = {".mp4", ".mkv", ".webm", ".mov"}
    USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    )
    _VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")
    _CHANNEL_PATH_HINTS = ("@", "channel/", "c/", "user/")

    def __init__(
        self,
        base_dir: Path,
        logger: logging.Logger,
        browser_priority: Optional[Sequence[str]] = None,
        maximum_quality_rank: Optional[int] = None,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.logger = logger
        self.browser_priority = tuple(browser_priority or self.DEFAULT_BROWSERS)
        self.maximum_quality_rank = maximum_quality_rank

        self._cookies_checked = False
        self._cookies_file: Optional[Path] = None
        self._browser_cookies: Optional[Tuple[str, ...]] = None
        self._download_history_cache: Optional[set[str]] = None
        self._fetched_history_cache: Optional[set[str]] = None
        self._js_runtimes = self._detect_js_runtimes()

        self.download_history_file.parent.mkdir(parents=True, exist_ok=True)
        self.fetched_history_file.parent.mkdir(parents=True, exist_ok=True)
        self.download_history_file.touch(exist_ok=True)
        self.fetched_history_file.touch(exist_ok=True)

    @staticmethod
    def _summarize_exception(exc: Exception, max_len: int = 260) -> str:
        text = " ".join(str(exc).split())
        if len(text) <= max_len:
            return text
        return f"{text[:max_len].rstrip()}..."

    @staticmethod
    def _detect_js_runtimes() -> Dict[str, Dict[str, Any]]:
        runtimes: Dict[str, Dict[str, Any]] = {}
        if shutil.which("node"):
            runtimes["node"] = {}
        if shutil.which("deno"):
            runtimes["deno"] = {}
        return runtimes

    def diagnostics(self) -> Dict[str, Any]:
        self._detect_cookies()
        return {
            "cookie_file": str(self._cookies_file) if self._cookies_file else None,
            "browser_cookie_source": self._browser_cookies[0] if self._browser_cookies else None,
            "js_runtimes": sorted(self._js_runtimes.keys()),
        }

    def probe_video_access(self, url: str = "https://www.youtube.com/watch?v=dQw4w9WgXcQ") -> Tuple[bool, str]:
        probe_opts: Dict[str, Any] = {
            "quiet": True,
            "skip_download": True,
            "extract_flat": False,
            "ignoreerrors": False,
            "no_warnings": True,
            "ignoreconfig": True,
            "logger": _SilentYDLLogger(),
        }
        self._apply_cookies_to_opts(probe_opts)
        if self._js_runtimes:
            probe_opts["js_runtimes"] = dict(self._js_runtimes)
            probe_opts["remote_components"] = ["ejs:github"]
        try:
            with yt_dlp.YoutubeDL(probe_opts) as ydl:
                ydl.extract_info(url, download=False)
            return True, "Probe succeeded."
        except Exception as exc:
            return False, self._summarize_exception(exc)

    @property
    def download_history_file(self) -> Path:
        return self.base_dir / "system" / "downloaded_youtube_links.txt"

    @property
    def fetched_history_file(self) -> Path:
        return self.base_dir / "system" / "fetched_youtube_links.txt"

    @classmethod
    def normalize_video_url(cls, raw_url: str) -> str:
        value = (raw_url or "").strip()
        if not value:
            return value

        if cls._VIDEO_ID_PATTERN.fullmatch(value):
            return f"https://www.youtube.com/watch?v={value}"

        try:
            parsed = urlparse(value)
        except Exception:
            return value

        netloc = parsed.netloc.lower()
        path = parsed.path or ""
        video_id: Optional[str] = None

        if "youtu.be" in netloc:
            candidate = path.strip("/").split("/")[0]
            if cls._VIDEO_ID_PATTERN.fullmatch(candidate):
                video_id = candidate
        elif "youtube.com" in netloc:
            if path == "/watch":
                query = parse_qs(parsed.query)
                candidate = (query.get("v") or [None])[0]
                if candidate and cls._VIDEO_ID_PATTERN.fullmatch(candidate):
                    video_id = candidate
            else:
                parts = [part for part in path.split("/") if part]
                if len(parts) >= 2 and parts[0] in {"shorts", "embed", "live"}:
                    candidate = parts[1]
                    if cls._VIDEO_ID_PATTERN.fullmatch(candidate):
                        video_id = candidate

        if video_id:
            return f"https://www.youtube.com/watch?v={video_id}"
        return value

    @classmethod
    def normalize_channel_url(cls, raw_url: str) -> str:
        value = (raw_url or "").strip().strip('"').strip("'")
        if not value:
            return ""

        if value.startswith("@"):
            value = f"https://www.youtube.com/{value}"
        elif "://" not in value:
            value = f"https://www.youtube.com/{value.lstrip('/')}"

        try:
            parsed = urlparse(value)
        except Exception:
            return ""

        netloc = parsed.netloc.lower()
        if "youtube.com" not in netloc and "youtu.be" not in netloc:
            return ""

        path = parsed.path.strip("/")
        if not path:
            return ""

        if path.startswith("watch"):
            return ""

        if path.startswith(cls._CHANNEL_PATH_HINTS):
            base = path
            if base.endswith("/videos"):
                base = base[:-7].rstrip("/")
            return f"https://www.youtube.com/{base}/videos"

        return ""

    def _load_download_history(self) -> set[str]:
        if self._download_history_cache is not None:
            return set(self._download_history_cache)

        history: set[str] = set()
        with self.download_history_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                normalized = self.normalize_video_url(line.strip())
                if normalized:
                    history.add(normalized)
        self._download_history_cache = set(history)
        return history

    def _add_to_download_history(self, url: str) -> None:
        normalized = self.normalize_video_url(url)
        if not normalized:
            return

        history = self._load_download_history()
        if normalized in history:
            return

        with self.download_history_file.open("a", encoding="utf-8") as handle:
            handle.write(f"{normalized}\n")
        history.add(normalized)
        self._download_history_cache = history

    def _load_fetched_history(self) -> set[str]:
        if self._fetched_history_cache is not None:
            return set(self._fetched_history_cache)

        history: set[str] = set()
        with self.fetched_history_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                normalized = self.normalize_video_url(line.strip())
                if normalized:
                    history.add(normalized)
        self._fetched_history_cache = set(history)
        return history

    def _add_to_fetched_history(self, url: str) -> None:
        normalized = self.normalize_video_url(url)
        if not normalized:
            return

        history = self._load_fetched_history()
        if normalized in history:
            return

        with self.fetched_history_file.open("a", encoding="utf-8") as handle:
            handle.write(f"{normalized}\n")
        history.add(normalized)
        self._fetched_history_cache = history

    def mark_links_as_fetched(self, links: Sequence[str]) -> None:
        for link in links:
            self._add_to_fetched_history(str(link))

    def is_already_fetched(self, url: str) -> bool:
        normalized = self.normalize_video_url(url)
        return normalized in self._load_fetched_history()

    def is_in_history(self, url: str) -> bool:
        normalized = self.normalize_video_url(url)
        return normalized in self._load_download_history()

    def _remove_urls_from_history_file(self, history_file: Path, urls: Sequence[str]) -> int:
        remove_set = {
            normalized
            for raw_url in urls
            if (normalized := self.normalize_video_url(str(raw_url)))
        }
        if not remove_set:
            return 0

        existing_lines: List[str] = []
        if history_file.exists():
            with history_file.open("r", encoding="utf-8") as handle:
                existing_lines = handle.readlines()

        retained_lines: List[str] = []
        removed_count = 0
        for line in existing_lines:
            normalized = self.normalize_video_url(line.strip())
            if normalized and normalized in remove_set:
                removed_count += 1
                continue
            retained_lines.append(line if line.endswith("\n") else f"{line}\n")

        with history_file.open("w", encoding="utf-8") as handle:
            handle.writelines(retained_lines)

        if history_file == self.download_history_file:
            self._download_history_cache = {
                normalized
                for line in retained_lines
                if (normalized := self.normalize_video_url(line.strip()))
            }
        elif history_file == self.fetched_history_file:
            self._fetched_history_cache = {
                normalized
                for line in retained_lines
                if (normalized := self.normalize_video_url(line.strip()))
            }
        return removed_count

    @staticmethod
    def _download_strategies() -> List[Dict[str, Any]]:
        return [
            {
                "name": "cookies-generic",
                "use_cookies": True,
            },
            {
                "name": "no-cookies-generic",
                "use_cookies": False,
            },
            {
                "name": "cookies-mobile-tv",
                "use_cookies": True,
                "extractor_args": {"youtube": {"player_client": ["android", "tv_downgraded"]}},
            },
            {
                "name": "no-cookies-mobile-tv",
                "use_cookies": False,
                "extractor_args": {"youtube": {"player_client": ["android", "tv_downgraded"]}},
            },
        ]

    def _configure_strategy_opts(self, ydl_opts: Dict[str, Any], strategy: Dict[str, Any]) -> Dict[str, Any]:
        configured = dict(ydl_opts)
        configured.pop("extractor_args", None)
        if strategy.get("extractor_args"):
            configured["extractor_args"] = strategy["extractor_args"]

        if strategy.get("use_cookies", True):
            self._apply_cookies_to_opts(configured)
        else:
            configured.pop("cookiefile", None)
            configured.pop("cookiesfrombrowser", None)
        return configured

    def probe_download_format(self, url: str) -> FormatProbeResult:
        """Verify that at least one normal download strategy exposes an acceptable stream."""
        normalized = self.normalize_video_url(url)
        if not normalized:
            return FormatProbeResult(acceptable=False, error="Invalid YouTube URL or video ID.")

        base_opts: Dict[str, Any] = {
            "quiet": True,
            "skip_download": True,
            "noprogress": True,
            "no_warnings": True,
            "http_headers": {
                "User-Agent": self.USER_AGENT,
                "Accept-Language": "en-us,en;q=0.5",
            },
            "socket_timeout": 60,
            "retries": 5,
            "ignoreerrors": False,
            "allow_unplayable_formats": False,
            "ignoreconfig": True,
            "noplaylist": True,
            "logger": _SilentYDLLogger(),
        }
        if self._js_runtimes:
            base_opts["js_runtimes"] = dict(self._js_runtimes)
            base_opts["remote_components"] = ["ejs:github"]

        last_error = "All format-probe strategies failed."
        for strategy in self._download_strategies():
            ydl_opts = self._configure_strategy_opts(base_opts, strategy)
            try:
                decision = self._probe_and_decide_format(normalized, ydl_opts)
            except Exception as exc:
                last_error = f"{strategy['name']}: {self._summarize_exception(exc)}"
                continue
            if decision.acceptable and decision.format_string:
                return FormatProbeResult(
                    acceptable=True,
                    strategy=str(strategy["name"]),
                    decision=decision,
                )
            last_error = f"{strategy['name']}: {decision.reason or 'no acceptable format found'}"

        return FormatProbeResult(acceptable=False, error=last_error)

    def _find_cookie_file(self) -> Optional[Path]:
        for candidate in (
            self.base_dir / "resources" / "cookies.txt",
            self.base_dir / "system" / "cookies.txt",
            self.base_dir / "cookies.txt",
        ):
            if candidate.exists() and candidate.stat().st_size > 0:
                return candidate
        return None

    def _browser_cookie_source_is_valid(self, browser: str) -> bool:
        test_opts: Dict[str, Any] = {
            "quiet": True,
            "skip_download": True,
            "extract_flat": "in_playlist",
            "playlist_items": "1",
            "cookiesfrombrowser": (browser,),
            "ignoreerrors": True,
            "no_warnings": True,
            "ignoreconfig": True,
            "logger": _SilentYDLLogger(),
        }
        try:
            with yt_dlp.YoutubeDL(test_opts) as ydl:
                result = ydl.extract_info("https://www.youtube.com/@YouTube/videos", download=False)
            if result and result.get("entries"):
                self.logger.info("Using browser cookies from: %s", browser)
                return True
        except Exception as exc:
            error_text = str(exc).lower()
            if "dpapi" in error_text or "decrypt" in error_text:
                self.logger.warning(
                    "Cannot read %s cookies while browser is open/encrypted. Skipping browser cookies.",
                    browser,
                )
                return False
        return False

    def _detect_cookies(self) -> None:
        if self._cookies_checked:
            return
        self._cookies_checked = True

        self._cookies_file = self._find_cookie_file()
        if self._cookies_file:
            self.logger.info("Using cookie file: %s", self._cookies_file)
            return

        for browser in self.browser_priority:
            if self._browser_cookie_source_is_valid(browser):
                self._browser_cookies = (browser,)
                return

        self.logger.warning("No cookies detected. Some YouTube videos may be unavailable.")

    def _apply_cookies_to_opts(self, ydl_opts: Dict[str, Any]) -> None:
        self._detect_cookies()
        ydl_opts.pop("cookiefile", None)
        ydl_opts.pop("cookiesfrombrowser", None)
        if self._cookies_file:
            ydl_opts["cookiefile"] = str(self._cookies_file)
        elif self._browser_cookies:
            ydl_opts["cookiesfrombrowser"] = self._browser_cookies

    @staticmethod
    def _resolve_channel_id(result: Dict[str, Any], channel_url: str) -> Optional[str]:
        candidates: List[str] = []
        for key in ("channel_id", "uploader_id", "id", "webpage_url", "channel_url"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                candidates.append(value.strip())
        candidates.append(channel_url)

        for candidate in candidates:
            if candidate.startswith("UC") and len(candidate) >= 20:
                return candidate
            if candidate.startswith("UU") and len(candidate) >= 20:
                return f"UC{candidate[2:]}"
            match = re.search(r"/channel/(UC[\w-]+)", candidate)
            if match:
                return match.group(1)
        return None

    def list_recent_channel_links(
        self,
        channels: Sequence[str],
        hours_limit: int = 24,
        playlistend: int = 15,
    ) -> List[str]:
        threshold = datetime.now().astimezone() - timedelta(hours=max(1, int(hours_limit)))
        seen: set[str] = set()
        recent_links: List[str] = []

        list_opts: Dict[str, Any] = {
            "quiet": True,
            "skip_download": True,
            "extract_flat": "in_playlist",
            "playlistend": 1,
            "ignoreerrors": True,
            "no_warnings": True,
            "socket_timeout": 30,
            "ignoreconfig": True,
            "logger": _SilentYDLLogger(),
        }
        self._apply_cookies_to_opts(list_opts)

        with yt_dlp.YoutubeDL(list_opts) as ydl:
            for channel_url in channels:
                normalized_channel = self.normalize_channel_url(str(channel_url))
                if not normalized_channel:
                    self.logger.warning("Skipping invalid channel URL: %s", channel_url)
                    continue
                try:
                    result = ydl.extract_info(normalized_channel, download=False)
                except Exception as exc:
                    self.logger.warning("Could not inspect channel %s: %s", normalized_channel, exc)
                    continue
                if not isinstance(result, dict):
                    continue

                channel_id = self._resolve_channel_id(result, normalized_channel)
                if not channel_id:
                    self.logger.warning("Unable to resolve channel ID for %s", normalized_channel)
                    continue

                rss_url = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
                try:
                    response = requests.get(
                        rss_url,
                        timeout=15,
                        headers={"User-Agent": self.USER_AGENT},
                    )
                except Exception as exc:
                    self.logger.warning("Failed RSS request for %s: %s", normalized_channel, exc)
                    continue
                if response.status_code != 200:
                    self.logger.warning(
                        "RSS request failed for %s (status %s)",
                        normalized_channel,
                        response.status_code,
                    )
                    continue

                count = 0
                entries = re.findall(r"<entry>.*?</entry>", response.text, flags=re.DOTALL)
                for entry in entries:
                    video_match = re.search(r"<yt:videoId>([^<]+)</yt:videoId>", entry)
                    published_match = re.search(r"<published>([^<]+)</published>", entry)
                    if not video_match or not published_match:
                        continue

                    try:
                        published = datetime.fromisoformat(
                            published_match.group(1).replace("Z", "+00:00")
                        ).astimezone()
                    except ValueError:
                        continue
                    if published < threshold:
                        break

                    url = self.normalize_video_url(
                        f"https://www.youtube.com/watch?v={video_match.group(1).strip()}"
                    )
                    if not url or url in seen:
                        continue

                    seen.add(url)
                    recent_links.append(url)
                    count += 1
                    if count >= max(1, int(playlistend)):
                        break

                if len(entries) >= 15 and count >= len(entries) and count < max(1, int(playlistend)):
                    self.logger.info(
                        "RSS feed reached its %s-item limit for %s; using detailed channel metadata "
                        "to complete the requested %s-hour window.",
                        len(entries),
                        normalized_channel,
                        hours_limit,
                    )
                    detailed_opts: Dict[str, Any] = {
                        "quiet": True,
                        "skip_download": True,
                        "extract_flat": False,
                        "playlistend": max(1, int(playlistend)),
                        "ignoreerrors": True,
                        "no_warnings": True,
                        "socket_timeout": 60,
                        "ignoreconfig": True,
                        "logger": _SilentYDLLogger(),
                    }
                    self._apply_cookies_to_opts(detailed_opts)
                    try:
                        with yt_dlp.YoutubeDL(detailed_opts) as detailed_ydl:
                            detailed = detailed_ydl.extract_info(normalized_channel, download=False)
                    except Exception as exc:
                        self.logger.warning(
                            "Detailed channel fallback failed for %s: %s",
                            normalized_channel,
                            exc,
                        )
                        continue

                    for item in (detailed or {}).get("entries", []):
                        if not isinstance(item, dict):
                            continue
                        timestamp = item.get("timestamp") or item.get("release_timestamp")
                        if timestamp is None:
                            continue
                        try:
                            published = datetime.fromtimestamp(float(timestamp)).astimezone()
                        except (TypeError, ValueError, OSError):
                            continue
                        if published < threshold:
                            break
                        raw_url = item.get("webpage_url") or item.get("url") or item.get("id") or ""
                        url = self.normalize_video_url(str(raw_url))
                        if not url or url in seen:
                            continue
                        seen.add(url)
                        recent_links.append(url)
                        count += 1
                        if count >= max(1, int(playlistend)):
                            break

        return recent_links

    def fetch_recent_links(
        self,
        channels: Sequence[str],
        hours_limit: int = 24,
        playlistend: int = 15,
        mark_fetched: bool = True,
        respect_fetched_history: bool = True,
    ) -> List[str]:
        downloaded_history = self._load_download_history()
        fetched_history = self._load_fetched_history() if respect_fetched_history else set()
        recent_links = self.list_recent_channel_links(
            channels=channels,
            hours_limit=hours_limit,
            playlistend=playlistend,
        )

        links_found: List[str] = []
        for url in recent_links:
            if url in downloaded_history or (respect_fetched_history and url in fetched_history):
                continue
            links_found.append(url)
            if mark_fetched:
                fetched_history.add(url)
                self._add_to_fetched_history(url)
        return links_found

    @staticmethod
    def _new_video_files(before: List[Path], after: List[Path]) -> List[Path]:
        known_paths = {str(path.resolve()) for path in before}
        created: List[Path] = []
        for path in after:
            if str(path.resolve()) in known_paths:
                continue
            if path.suffix.lower() not in YTHandler.VIDEO_FILE_EXTENSIONS:
                continue
            if path.suffix.lower() in {".part", ".ytdl"}:
                continue
            if path.is_file():
                created.append(path)
        return created

    def _probe_video_info(self, target_url: str) -> Dict[str, Any]:
        probe_opts: Dict[str, Any] = {
            "skip_download": True,
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "ignoreconfig": True,
            "logger": _SilentYDLLogger(),
        }
        if self._js_runtimes:
            probe_opts["js_runtimes"] = dict(self._js_runtimes)
            probe_opts["remote_components"] = ["ejs:github"]

        last_error = "Unable to probe video metadata."
        for strategy in self._download_strategies():
            strategy_opts = self._configure_strategy_opts(probe_opts, strategy)
            try:
                with yt_dlp.YoutubeDL(strategy_opts) as ydl:
                    info = ydl.extract_info(target_url, download=False)
                if isinstance(info, dict):
                    return info
            except Exception as exc:
                last_error = f"{strategy['name']}: {self._summarize_exception(exc)}"
                continue

        raise RuntimeError(last_error)

    @staticmethod
    def _probe_media_file(path: Path) -> Dict[str, Any]:
        command = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,codec_name",
            "-of",
            "json",
            str(path),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        payload = json.loads(result.stdout or "{}")
        streams = payload.get("streams", [])
        if not isinstance(streams, list) or not streams:
            raise RuntimeError("No video stream found in downloaded file.")

        stream = streams[0] if isinstance(streams[0], dict) else {}
        width = int(stream.get("width") or 0)
        height = int(stream.get("height") or 0)
        codec = str(stream.get("codec_name") or "")
        return {"width": width, "height": height, "codec": codec}

    @staticmethod
    def _quality_rank_from_dimensions(width: int, height: int) -> tuple[int, str]:
        return classify_quality(width=width, height=height, format_note="")

    def _validate_downloaded_file(
        self,
        downloaded: Path,
        decision: FormatDecision,
    ) -> Tuple[bool, Dict[str, Any], str]:
        try:
            media = self._probe_media_file(downloaded)
        except Exception as exc:
            return False, {"width": 0, "height": 0, "codec": ""}, self._summarize_exception(exc)

        actual_rank, actual_label = self._quality_rank_from_dimensions(
            width=int(media.get("width", 0)),
            height=int(media.get("height", 0)),
        )
        media["quality_label"] = actual_label
        media["quality_rank"] = actual_rank
        if actual_rank < QUALITY_1080:
            return False, media, "Downloaded file is below the 1080-class minimum quality floor."
        if actual_rank < int(decision.quality_rank):
            return (
                False,
                media,
                f"Downloaded file did not meet the selected {decision.quality_label} quality class.",
            )
        return True, media, ""

    def _probe_and_decide_format(self, target_url: str, ydl_opts: Dict[str, Any]) -> FormatDecision:
        probe_opts = dict(ydl_opts)
        probe_opts.update(
            {
                "skip_download": True,
                "quiet": True,
                "no_warnings": True,
                "noplaylist": True,
            }
        )
        with yt_dlp.YoutubeDL(probe_opts) as ydl:
            info = ydl.extract_info(target_url, download=False)
        if not isinstance(info, dict):
            return FormatDecision(
                format_string=None,
                strategy="probe-no-info",
                acceptable=False,
                quality_rank=0,
                quality_label="below-1080",
                reason="Probe did not return video metadata.",
            )
        return choose_download_format(info, maximum_quality_rank=self.maximum_quality_rank)

    def _matching_local_video_files(
        self,
        url: str,
        directories: Sequence[Path],
        filename_template: str,
    ) -> List[Path]:
        try:
            info = self._probe_video_info(url)
        except Exception as exc:
            self.logger.warning("Failed to resolve local filenames for %s: %s", url, exc)
            return []

        try:
            with yt_dlp.YoutubeDL({"quiet": True, "ignoreconfig": True, "outtmpl": filename_template}) as ydl:
                expected_stem = Path(ydl.prepare_filename({**info, "ext": "mp4"})).stem
        except Exception as exc:
            self.logger.warning("Failed to prepare expected filename for %s: %s", url, exc)
            return []

        matches: List[Path] = []
        for directory in directories:
            for path in Path(directory).glob(f"{expected_stem}.*"):
                if path.suffix.lower() not in self.VIDEO_FILE_EXTENSIONS:
                    continue
                if path.is_file():
                    matches.append(path)
        return matches

    def reset_recent_channel_history(
        self,
        *,
        channels: Sequence[str],
        hours_limit: int,
        playlistend: int = 15,
        replace_dirs: Sequence[Path] = (),
        filename_template: str = "%(title).200B.%(ext)s",
    ) -> Dict[str, Any]:
        recent_links = self.list_recent_channel_links(
            channels=channels,
            hours_limit=hours_limit,
            playlistend=playlistend,
        )
        removed_downloaded = self._remove_urls_from_history_file(self.download_history_file, recent_links)
        removed_fetched = self._remove_urls_from_history_file(self.fetched_history_file, recent_links)

        removed_files: List[str] = []
        for url in recent_links:
            for path in self._matching_local_video_files(url, directories=replace_dirs, filename_template=filename_template):
                try:
                    path.unlink(missing_ok=True)
                except Exception as exc:
                    self.logger.warning("Failed to remove old source %s: %s", path, exc)
                    continue
                removed_files.append(str(path))

        summary = {
            "recent_links": recent_links,
            "removed_from_download_history": removed_downloaded,
            "removed_from_fetched_history": removed_fetched,
            "removed_local_files": removed_files,
        }
        self.logger.info(
            "Recent YouTube history reset completed: %s links, %s downloaded-history entries, "
            "%s fetched-history entries, %s local files removed.",
            len(recent_links),
            removed_downloaded,
            removed_fetched,
            len(removed_files),
        )
        return summary

    def download_video(
        self,
        url: str,
        output_dir: Path,
        filename_template: str = "%(title).200B.%(ext)s",
    ) -> DownloadResult:
        normalized = self.normalize_video_url(url)
        if self.is_in_history(normalized):
            self.logger.info("Skipping already downloaded URL: %s", normalized)
            return DownloadResult(status="already_downloaded")

        target_url = normalized or url
        output_dir.mkdir(parents=True, exist_ok=True)
        before_files = list(output_dir.glob("*"))

        base_opts: Dict[str, Any] = {
            "outtmpl": str(output_dir / filename_template),
            "quiet": True,
            "noprogress": True,
            "no_warnings": True,
            "http_headers": {
                "User-Agent": self.USER_AGENT,
                "Accept-Language": "en-us,en;q=0.5",
            },
            "socket_timeout": 60,
            "retries": 15,
            "fragment_retries": 15,
            "skip_unavailable_fragments": True,
            "ignoreerrors": False,
            "allow_unplayable_formats": False,
            "ignoreconfig": True,
            "noplaylist": True,
            "logger": _SilentYDLLogger(),
        }
        if self._js_runtimes:
            base_opts["js_runtimes"] = dict(self._js_runtimes)
            base_opts["remote_components"] = ["ejs:github"]

        last_error = "All download strategies failed."
        for strategy in self._download_strategies():
            ydl_opts = self._configure_strategy_opts(base_opts, strategy)

            try:
                decision = self._probe_and_decide_format(target_url, ydl_opts)
            except Exception as exc:
                error_summary = self._summarize_exception(exc)
                self.logger.warning(
                    "Format probe failed for strategy '%s' on %s: %s",
                    strategy["name"],
                    target_url,
                    error_summary,
                )
                last_error = f"{strategy['name']}: {error_summary}"
                continue

            if not decision.acceptable or not decision.format_string:
                self.logger.info(
                    "Skipping strategy '%s' for %s because best result was rejected: %s "
                    "(format_id=%s, %sx%s, vcodec=%s, source=%s)",
                    strategy["name"],
                    target_url,
                    decision.reason or decision.strategy,
                    decision.format_id or "unknown",
                    decision.width,
                    decision.height,
                    decision.vcodec or "unknown",
                    decision.source_type or "unknown",
                )
                last_error = f"{strategy['name']}: {decision.reason or 'no acceptable format found'}"
                continue

            ydl_opts["format"] = str(decision.format_string)

            self.logger.info(
                "Downloading with %s (%s): %s [format_id=%s, %sx%s, vcodec=%s, source=%s]",
                strategy["name"],
                decision.strategy,
                target_url,
                decision.format_id or "unknown",
                decision.width,
                decision.height,
                decision.vcodec or "unknown",
                decision.source_type or "unknown",
            )
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    code = ydl.download([target_url])
            except Exception as exc:
                error_summary = self._summarize_exception(exc)
                self.logger.warning(
                    "Download strategy '%s' failed for %s: %s",
                    strategy["name"],
                    target_url,
                    error_summary,
                )
                last_error = f"{strategy['name']}: {error_summary}"
                continue

            if code not in (None, 0):
                self.logger.warning(
                    "yt-dlp returned non-zero status %s for %s using '%s'",
                    code,
                    target_url,
                    strategy["name"],
                )
                last_error = f"{strategy['name']}: non-zero yt-dlp status {code}"
                continue

            created_files = self._new_video_files(before_files, list(output_dir.glob("*")))
            if not created_files:
                self.logger.warning("No file created for %s with strategy '%s'", target_url, strategy["name"])
                last_error = f"{strategy['name']}: no output file created"
                continue

            created_files.sort(key=lambda path: path.stat().st_mtime, reverse=True)
            downloaded = created_files[0]
            size = downloaded.stat().st_size
            if size < 10_000:
                self.logger.warning("Downloaded file too small (%s bytes): %s", size, downloaded.name)
                downloaded.unlink(missing_ok=True)
                last_error = f"{strategy['name']}: downloaded file too small ({size} bytes)"
                continue

            valid_file, media, validation_error = self._validate_downloaded_file(downloaded, decision)
            self.logger.info(
                "Validated downloaded file %s: %sx%s %s (%s)",
                downloaded.name,
                int(media.get("width", 0)),
                int(media.get("height", 0)),
                str(media.get("codec", "") or "unknown"),
                str(media.get("quality_label", "unknown")),
            )
            if not valid_file:
                self.logger.warning(
                    "Discarding %s from strategy '%s': %s",
                    downloaded.name,
                    strategy["name"],
                    validation_error,
                )
                downloaded.unlink(missing_ok=True)
                last_error = f"{strategy['name']}: {validation_error}"
                continue

            self._add_to_download_history(target_url)
            self.logger.info("Download succeeded for %s (%s)", target_url, downloaded.name)
            return DownloadResult(
                status="downloaded",
                path=str(downloaded),
                strategy=strategy["name"],
            )

        return DownloadResult(status="failed", error=last_error)
