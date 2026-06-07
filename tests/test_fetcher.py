"""Unit tests for src/fetcher.py bug fixes (hermetic, no network)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.fetcher import _parse_relative_time


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
