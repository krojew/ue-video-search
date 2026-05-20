"""Fetch video metadata from a YouTube channel using scrapetube."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
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
    for unit, delta_fn in [
        ("year", lambda n: timedelta(days=n * 365)),
        ("month", lambda n: timedelta(days=n * 30)),
        ("week", lambda n: timedelta(weeks=n)),
        ("day", lambda n: timedelta(days=n)),
        ("hour", lambda n: timedelta(hours=n)),
    ]:
        if unit in text:
            # Extract the first integer token (skips leading words like "Streamed")
            for token in text.split():
                try:
                    num = int(token)
                    return now - delta_fn(num)
                except ValueError:
                    continue
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
    yt_dlp_bin = shutil.which("yt-dlp") or str(Path(sys.executable).parent / "yt-dlp")

    # The /streams URL surfaces the "Live" tab on YouTube channels.
    base = channel_url.rstrip("/")
    urls = [base]
    if include_streams:
        urls.append(f"{base}/streams")

    converted: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for url in urls:
        try:
            result = subprocess.run(
                [
                    yt_dlp_bin,
                    "--no-warnings",
                    "--flat-playlist",
                    "--dump-single-json",
                    url,
                ],
                capture_output=True,
                text=True,
                check=True,
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
            duration_secs = entry.get("duration")
            if duration_secs is None:
                continue

            upload_date = entry.get("upload_date")
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

    # scrapetube yields raw video renderer dicts from the channel page.
    # Optionally fetch from the "Live" tab to include past streams.
    content_types = ["videos"]
    if include_streams:
        content_types.append("streams")

    raw_videos: list[dict[str, Any]] = []
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

    if not raw_videos:
        raw_videos = _fetch_channel_videos_with_yt_dlp(
            CHANNEL_URL, include_streams=include_streams
        )

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

        title_lower = title.lower()

        # Skip Unreal Editor for Fortnite content
        if skip_uefn and ("uefn" in title_lower or "fortnite" in title_lower):
            continue

        if skip_automotive and "automotive" in title_lower:
            continue

        if skip_archvis and "archvis" in title_lower:
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

        # Publish date (relative)
        pub_text = v.get("publishedTimeText", {}).get("simpleText", "")
        pub_date = _parse_relative_time(pub_text) if pub_text else None
        if pub_date is None and v.get("upload_date"):
            try:
                pub_date = datetime.strptime(v["upload_date"], "%Y%m%d").replace(tzinfo=timezone.utc)
                pub_text = pub_text or _format_relative_time(pub_date)
            except ValueError:
                pub_date = None

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
    existing_ids = {v["video_id"] for v in existing}
    new_only = [v for v in incoming if v["video_id"] not in existing_ids]

    # Build merged list: new videos first (newest), then existing
    merged_ids: set[str] = set()
    merged: list[dict[str, Any]] = []
    for v in incoming + existing:
        if v["video_id"] not in merged_ids:
            merged_ids.add(v["video_id"])
            merged.append(v)

    return merged, new_only


def save_video_list(videos: list[dict[str, Any]], path: Path | None = None) -> Path:
    """Persist the video list to JSON."""
    path = path or DATA_DIR / "videos.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(videos, indent=2, default=str))
    return path


def load_video_list(path: Path | None = None) -> list[dict[str, Any]]:
    """Load a previously saved video list."""
    path = path or DATA_DIR / "videos.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())
