"""Fetch video metadata from a YouTube channel using yt-dlp (scrapetube fallback)."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import scrapetube

from .config import (
    CHANNEL_URL,
    DATA_DIR,
    MAX_AGE_YEARS,
    MIN_DURATION_SECONDS,
)


_UEFN_RE = re.compile(r"\b(uefn|fortnite)\b", re.IGNORECASE)
_AUTOMOTIVE_RE = re.compile(r"\bautomotive\b", re.IGNORECASE)
_ARCHVIS_RE = re.compile(r"\barchvi[sz]\b", re.IGNORECASE)


def _parse_duration_text(text: str) -> int | None:
    """Convert duration strings like '1:23:45' or '23:45' to total seconds.

    Returns None for non-numeric values (e.g. 'Upcoming' for scheduled streams).
    """
    parts = text.strip().split(":")
    try:
        parts = [int(p) for p in parts]
    except ValueError:
        return None
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0]


def _parse_relative_time(text: str) -> datetime | None:
    """Best-effort parse of YouTube relative timestamps.

    Handles both regular uploads ("2 years ago") and streams
    ("Streamed 2 years ago").
    """
    text = text.lower().strip()
    now = datetime.now(timezone.utc)

    # "just now" / "a moment ago" -> treat as the current instant.
    if "just now" in text or "moment" in text:
        return now

    for unit, delta_fn in [
        ("year", lambda n: timedelta(days=n * 365)),
        ("month", lambda n: timedelta(days=n * 30)),
        ("week", lambda n: timedelta(weeks=n)),
        ("day", lambda n: timedelta(days=n)),
        ("hour", lambda n: timedelta(hours=n)),
        ("minute", lambda n: timedelta(minutes=n)),
        ("second", lambda n: timedelta(seconds=n)),
    ]:
        if unit in text:
            # Extract the first integer token (skips leading words like "Streamed")
            for token in text.split():
                try:
                    num = int(token)
                    return now - delta_fn(num)
                except ValueError:
                    continue
            # No integer token found, but a unit word is present. YouTube uses
            # "a year ago" / "an hour ago" for a single unit -> treat as 1.
            if "a" in text.split() or "an" in text.split():
                return now - delta_fn(1)
            return None

    # Support ISO-style dates in fallback mode
    try:
        return datetime.fromisoformat(text).astimezone(timezone.utc)
    except ValueError:
        return None


def _format_relative_time(dt: datetime) -> str:
    """Format datetime as a human-friendly relative time string."""
    now = datetime.now(timezone.utc)
    delta = now - dt
    if delta.days >= 365:
        years = delta.days // 365
        return f"{years} year{'s' if years != 1 else ''} ago"
    if delta.days >= 30:
        months = delta.days // 30
        return f"{months} month{'s' if months != 1 else ''} ago"
    if delta.days >= 7:
        weeks = delta.days // 7
        return f"{weeks} week{'s' if weeks != 1 else ''} ago"
    if delta.days >= 1:
        return f"{delta.days} day{'s' if delta.days != 1 else ''} ago"
    hours = delta.seconds // 3600
    if hours >= 1:
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    minutes = delta.seconds // 60
    return f"{minutes} minute{'s' if minutes != 1 else ''} ago"


def _format_duration(seconds: int | None) -> str:
    """Format a duration in seconds as H:MM:SS or M:SS."""
    if seconds is None:
        return ""
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02}:{secs:02}"
    return f"{minutes}:{secs:02}"


def _fetch_channel_videos_with_yt_dlp(
    channel_url: str,
    include_streams: bool = True,
) -> list[dict[str, Any]]:
    """Fallback channel video listing using yt-dlp JSON output.

    Fetches from both the default videos tab and the live/streams tab.
    """
    # Prefer the installed `yt-dlp` console script; fall back to the package's
    # __main__ via the running interpreter. The latter works even when the
    # script directory is not on PATH (common on Windows pip --user installs).
    yt_dlp_cmd: list[str]
    yt_dlp_path = shutil.which("yt-dlp") or shutil.which("yt-dlp.exe")
    if yt_dlp_path:
        yt_dlp_cmd = [yt_dlp_path]
    else:
        yt_dlp_cmd = [sys.executable, "-m", "yt_dlp"]

    # Drill into the per-tab URLs. yt-dlp on a bare channel URL returns
    # a playlist of tabs (Videos/Live/Shorts) rather than videos themselves,
    # so the /videos and /streams suffixes are required to enumerate items.
    base = channel_url.rstrip("/")
    urls = [f"{base}/videos"]
    if include_streams:
        urls.append(f"{base}/streams")

    converted: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for url in urls:
        try:
            result = subprocess.run(
                [
                    *yt_dlp_cmd,
                    "--no-warnings",
                    "--flat-playlist",
                    "--dump-single-json",
                    # `approximate_date` populates the per-entry `timestamp`
                    # field during a flat-playlist scrape. Without it,
                    # upload dates are only available via per-video extraction
                    # (~1s/video).
                    "--extractor-args",
                    "youtubetab:approximate_date",
                    url,
                ],
                capture_output=True,
                text=True,
                check=True,
                # Bound the fetch so a hung yt-dlp surfaces as
                # TimeoutExpired (caught below) and this tab is skipped
                # instead of blocking the whole run forever.
                timeout=120,
            )
            payload = json.loads(result.stdout)
        except Exception:
            continue

        entries = payload.get("entries", []) or []
        for entry in entries:
            video_id = entry.get("id")
            if not video_id or video_id in seen_ids:
                continue
            seen_ids.add(video_id)

            title = entry.get("title", "")
            duration_raw = entry.get("duration")
            if duration_raw is None:
                continue
            try:
                duration_secs = int(duration_raw)
            except (TypeError, ValueError):
                continue

            upload_date = entry.get("upload_date")
            timestamp = entry.get("timestamp")
            if not upload_date and timestamp is not None:
                try:
                    upload_date = datetime.fromtimestamp(
                        timestamp, tz=timezone.utc
                    ).strftime("%Y%m%d")
                except (OverflowError, OSError, ValueError):
                    upload_date = None

            published_text = ""
            if upload_date:
                try:
                    date = datetime.strptime(upload_date, "%Y%m%d").replace(tzinfo=timezone.utc)
                    published_text = _format_relative_time(date)
                except ValueError:
                    published_text = upload_date

            converted.append(
                {
                    "videoId": video_id,
                    "title": {"runs": [{"text": title}]},
                    "lengthText": {"simpleText": _format_duration(duration_secs)},
                    "publishedTimeText": {"simpleText": published_text},
                    "upload_date": upload_date,
                }
            )
    return converted


def fetch_video_list(
    skip_uefn: bool = True,
    skip_automotive: bool = True,
    skip_archvis: bool = True,
    include_streams: bool = True,
) -> list[dict[str, Any]]:
    """Return metadata dicts for qualifying videos from the channel.

    Filters applied:
      - Published within the last MAX_AGE_YEARS years
      - Duration >= MIN_DURATION_SECONDS
      - Title exclusion filters for UEFN/Fortnite, automotive, and archvis
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=MAX_AGE_YEARS * 365)

    # yt-dlp is the primary source: it reliably enumerates the full channel
    # (including past livestreams from the /streams tab) and tracks YouTube's
    # page-format changes closely. scrapetube is faster but periodically
    # returns a silent empty/partial result when YouTube alters its internal
    # JSON layout, which would otherwise drop the freshest videos. We only
    # fall back to scrapetube if yt-dlp yields nothing.
    raw_videos: list[dict[str, Any]] = _fetch_channel_videos_with_yt_dlp(
        CHANNEL_URL, include_streams=include_streams
    )

    if not raw_videos:
        content_types = ["videos"]
        if include_streams:
            content_types.append("streams")
        for content_type in content_types:
            try:
                raw_videos.extend(
                    scrapetube.get_channel(
                        channel_url=CHANNEL_URL,
                        sort_by="newest",
                        content_type=content_type,
                    )
                )
            except Exception:
                pass

    results: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for v in raw_videos:
        video_id = v.get("videoId", "")
        if not video_id or video_id in seen_ids:
            continue
        seen_ids.add(video_id)
        title_runs = v.get("title", {}).get("runs", [])
        if title_runs:
            title = "".join(run.get("text", "") for run in title_runs)
        else:
            title = v.get("title", {}).get("simpleText", "")

        if skip_uefn and _UEFN_RE.search(title):
            continue

        if skip_automotive and _AUTOMOTIVE_RE.search(title):
            continue

        if skip_archvis and _ARCHVIS_RE.search(title):
            continue

        # Duration
        duration_text = (
            v.get("lengthText", {}).get("simpleText", "")
            or v.get("thumbnailOverlays", [{}])[0]
            .get("thumbnailOverlayTimeStatusRenderer", {})
            .get("text", {})
            .get("simpleText", "")
        )
        if not duration_text:
            continue
        duration_secs = _parse_duration_text(duration_text)
        if duration_secs is None or duration_secs < MIN_DURATION_SECONDS:
            continue

        # Publish date. Prefer the precise `upload_date` (YYYYMMDD) when the
        # yt-dlp path provides it, since the relative text ("2 days ago") loses
        # precision and can flip include/exclude decisions near the cutoff.
        # The scrapetube path only has relative text, so fall back to that.
        pub_text = v.get("publishedTimeText", {}).get("simpleText", "")
        pub_date = None
        if v.get("upload_date"):
            try:
                pub_date = datetime.strptime(
                    v["upload_date"], "%Y%m%d"
                ).replace(tzinfo=timezone.utc)
                # upload_date is authoritative here, so derive the displayed
                # relative text from it too — otherwise a stale/mismatched
                # publishedTimeText could disagree with published_date.
                pub_text = _format_relative_time(pub_date)
            except (ValueError, TypeError):
                pub_date = None
        if pub_date is None and pub_text:
            pub_date = _parse_relative_time(pub_text)

        if pub_date and pub_date < cutoff:
            continue
        if pub_date is None:
            # If we can't parse the date, skip to be safe
            continue

        results.append(
            {
                "video_id": video_id,
                "title": title,
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "duration_seconds": duration_secs,
                "duration_text": duration_text,
                "published_text": pub_text,
                "published_date": pub_date.isoformat(),
            }
        )

    return results


def merge_video_lists(
    existing: list[dict[str, Any]],
    incoming: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Merge incoming videos into the existing list, deduplicating by video_id.

    Returns (merged_list, new_only) where new_only contains videos
    that were not in the existing list.
    """
    existing_ids = {
        v["video_id"] for v in existing if v.get("video_id")
    }
    new_only = [
        v for v in incoming
        if v.get("video_id") and v["video_id"] not in existing_ids
    ]

    # Build merged list: new videos first (newest), then existing.
    # Entries lacking a truthy "video_id" are skipped rather than aborting
    # the whole merge with a KeyError.
    merged_ids: set[str] = set()
    merged: list[dict[str, Any]] = []
    for v in incoming + existing:
        vid = v.get("video_id")
        if not vid or vid in merged_ids:
            continue
        merged_ids.add(vid)
        merged.append(v)

    return merged, new_only


def save_video_list(videos: list[dict[str, Any]], path: Path | None = None) -> Path:
    """Persist the video list to JSON atomically.

    Writes to a temp file in the same directory and then ``os.replace()``s it
    into place, which is atomic on POSIX. This guarantees that a crash mid-write
    leaves the previous cache intact rather than a truncated/corrupt file.
    """
    path = path or DATA_DIR / "videos.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(videos, indent=2, default=str)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=path.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        # Clean up the temp file if anything went wrong before the replace.
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return path


def save_fetch_result(
    fresh: list[dict[str, Any]],
    path: Path | None = None,
) -> list[dict[str, Any]]:
    """Persist a freshly-fetched video list, guarding against bad fetches.

    A transient yt-dlp/scrapetube hiccup can return an empty or heavily
    truncated list. Blindly overwriting the cache with it would destroy the
    good on-disk history, and a subsequent ``purge`` would then treat the
    truncated list as the allow-set and delete valid indexed videos from
    Qdrant. So we refuse to overwrite when a populated cache already exists and
    the fresh result is empty or less than half its size; in that case the
    existing cache is kept and returned unchanged.

    Returns the list that is now authoritative on disk (either ``fresh`` after
    a successful save, or the retained ``cached`` list).
    """
    cached = load_video_list(path)
    if cached and (not fresh or len(fresh) < len(cached) * 0.5):
        return cached
    save_video_list(fresh, path)
    return fresh


def load_video_list(path: Path | None = None) -> list[dict[str, Any]]:
    """Load a previously saved video list.

    Tolerates a missing, empty, or corrupt cache file by returning an empty
    list. A genuine I/O error (e.g. permission denied) on an existing file is
    NOT swallowed — it propagates, so callers don't mistake an unreadable cache
    for an empty one and then overwrite the good history.
    """
    path = path or DATA_DIR / "videos.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, ValueError):
        return []
