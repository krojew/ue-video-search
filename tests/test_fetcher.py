"""Unit tests for src/fetcher.py bug fixes (hermetic, no network)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src import fetcher
from src.fetcher import _parse_relative_time, fetch_video_list


# ── BUG 1: _parse_relative_time drops fresh/single-unit videos ──────────────


@pytest.mark.parametrize(
    "text, max_age",
    [
        ("3 minutes ago", timedelta(minutes=10)),
        ("30 seconds ago", timedelta(minutes=5)),
        ("just now", timedelta(minutes=1)),
        ("a year ago", timedelta(days=400)),
        ("an hour ago", timedelta(hours=3)),
    ],
)
def test_parse_relative_time_fresh_and_single_unit(text, max_age):
    """Fresh/single-unit timestamps must parse to a non-None recent datetime."""
    now = datetime.now(timezone.utc)
    parsed = _parse_relative_time(text)
    assert parsed is not None, f"{text!r} parsed to None"
    assert parsed.tzinfo is not None  # timezone-aware
    # Parsed time is in the past (or now) and no older than the expected window.
    assert parsed <= now + timedelta(seconds=2)
    assert now - parsed <= max_age


def test_parse_relative_time_a_year_is_about_one_year():
    now = datetime.now(timezone.utc)
    parsed = _parse_relative_time("a year ago")
    delta_days = (now - parsed).days
    assert 360 <= delta_days <= 370


def test_parse_relative_time_an_hour_is_about_one_hour():
    now = datetime.now(timezone.utc)
    parsed = _parse_relative_time("an hour ago")
    assert timedelta(minutes=59) <= (now - parsed) <= timedelta(minutes=61)


def test_parse_relative_time_streamed_prefix_still_works():
    now = datetime.now(timezone.utc)
    parsed = _parse_relative_time("Streamed 2 years ago")
    assert parsed is not None
    assert 720 <= (now - parsed).days <= 740


def test_parse_relative_time_cutoff_ordering():
    """A 5-year-old video is older than a 3-year cutoff; 2 days is newer."""
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=3 * 365)

    old = _parse_relative_time("5 years ago")
    recent = _parse_relative_time("2 days ago")
    assert old is not None and recent is not None
    assert old < cutoff
    assert recent > cutoff


# ── BUG 2: cutoff prefers precise upload_date over relative text ────────────


def _fake_yt_dlp_factory(entries):
    def _fake(channel_url, include_streams=True):
        return entries

    return _fake


def test_fetch_video_list_prefers_precise_upload_date(monkeypatch):
    """When upload_date is present it must win over the relative text.

    The relative text claims "2 years ago" (well inside a 3-year cutoff) but
    upload_date pins the video to 20180101 (outside the cutoff). The precise
    date must win, so the video is excluded.
    """
    raw = [
        {
            "videoId": "OLD123",
            "title": {"runs": [{"text": "Ancient Tutorial"}]},
            "lengthText": {"simpleText": "30:00"},
            "publishedTimeText": {"simpleText": "2 years ago"},
            "upload_date": "20180101",
        },
    ]
    monkeypatch.setattr(
        fetcher, "_fetch_channel_videos_with_yt_dlp", _fake_yt_dlp_factory(raw)
    )
    results = fetch_video_list(include_streams=False)
    assert results == [], "precise old upload_date should exclude the video"


def test_fetch_video_list_upload_date_drives_published_date(monkeypatch):
    """published_date must reflect upload_date, not the re-parsed relative text."""
    recent = (datetime.now(timezone.utc) - timedelta(days=10)).strftime("%Y%m%d")
    raw = [
        {
            "videoId": "NEW456",
            "title": {"runs": [{"text": "Recent Tutorial"}]},
            "lengthText": {"simpleText": "30:00"},
            # Deliberately mismatched relative text to prove precedence.
            "publishedTimeText": {"simpleText": "2 years ago"},
            "upload_date": recent,
        },
    ]
    monkeypatch.setattr(
        fetcher, "_fetch_channel_videos_with_yt_dlp", _fake_yt_dlp_factory(raw)
    )
    results = fetch_video_list(include_streams=False)
    assert len(results) == 1
    pub = datetime.fromisoformat(results[0]["published_date"])
    expected = datetime.strptime(recent, "%Y%m%d").replace(tzinfo=timezone.utc)
    assert pub == expected


def test_fetch_video_list_scrapetube_relative_text_still_works(monkeypatch):
    """Entries without upload_date (scrapetube path) fall back to relative text."""
    raw = [
        {
            "videoId": "REL789",
            "title": {"runs": [{"text": "Stream Replay"}]},
            "lengthText": {"simpleText": "45:00"},
            "publishedTimeText": {"simpleText": "2 days ago"},
            # no upload_date key at all
        },
    ]
    monkeypatch.setattr(
        fetcher, "_fetch_channel_videos_with_yt_dlp", _fake_yt_dlp_factory(raw)
    )
    results = fetch_video_list(include_streams=False)
    assert len(results) == 1
    assert results[0]["video_id"] == "REL789"
