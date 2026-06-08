"""Unit tests for src/fetcher.py bug fixes (hermetic, no network)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

import json
import subprocess

from src import fetcher
from src.fetcher import (
    _parse_relative_time,
    fetch_video_list,
    load_video_list,
    merge_video_lists,
    save_fetch_result,
    save_video_list,
)


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
    # published_text must be regenerated from upload_date too, not left as the
    # mismatched "2 years ago" — the two fields must not diverge.
    assert "year" not in results[0]["published_text"], results[0]["published_text"]
    assert results[0]["published_text"] == fetcher._format_relative_time(expected)


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


# ── BUG 3: atomic save + tolerant load ──────────────────────────────────────


def test_save_load_round_trip(tmp_path):
    path = tmp_path / "videos.json"
    videos = [
        {"video_id": "abc", "title": "One"},
        {"video_id": "def", "title": "Two"},
    ]
    returned = save_video_list(videos, path=path)
    assert returned == path
    assert path.exists()
    loaded = load_video_list(path=path)
    assert loaded == videos


def test_load_corrupt_file_returns_empty(tmp_path):
    path = tmp_path / "videos.json"
    path.write_text("{ this is not valid json :::")
    assert load_video_list(path=path) == []


def test_load_empty_file_returns_empty(tmp_path):
    path = tmp_path / "videos.json"
    path.write_text("")
    assert load_video_list(path=path) == []


def test_load_missing_file_returns_empty(tmp_path):
    path = tmp_path / "does_not_exist.json"
    assert load_video_list(path=path) == []


def test_save_leaves_no_temp_files(tmp_path):
    path = tmp_path / "videos.json"
    save_video_list([{"video_id": "x"}], path=path)
    # Only the final file should remain — no leftover *.tmp artifacts.
    leftovers = [p.name for p in tmp_path.iterdir() if p.name != "videos.json"]
    assert leftovers == []


# ── BUG 4: merge_video_lists skips entries missing video_id ──────────────────


def test_merge_skips_entries_missing_video_id():
    existing = [
        {"video_id": "a", "title": "A"},
        {"title": "no id in existing"},  # missing key
    ]
    incoming = [
        {"video_id": "b", "title": "B"},
        {"title": "no id in incoming"},  # missing key
        {"video_id": "", "title": "empty id"},  # falsy id
        {"video_id": "a", "title": "A dup"},  # already existing
    ]
    merged, new_only = merge_video_lists(existing, incoming)

    merged_ids = [v.get("video_id") for v in merged]
    # No KeyError, and only truthy-id entries survive.
    assert "a" in merged_ids
    assert "b" in merged_ids
    assert "" not in merged_ids
    assert None not in merged_ids
    # new_only contains only genuinely new, truthy-id entries.
    assert [v["video_id"] for v in new_only] == ["b"]


def test_merge_empty_inputs():
    merged, new_only = merge_video_lists([], [])
    assert merged == []
    assert new_only == []


# ── R1: save_fetch_result guard (shared by run_fetch and the web worker) ─────


def test_save_fetch_result_keeps_cache_on_empty_fetch(tmp_path):
    """An empty fetch must NOT overwrite a populated cache; cache is returned."""
    path = tmp_path / "videos.json"
    cached = [{"video_id": c} for c in "abcde"]
    save_video_list(cached, path=path)

    result = save_fetch_result([], path=path)

    assert [v["video_id"] for v in result] == list("abcde")
    # On-disk cache is untouched.
    assert [v["video_id"] for v in load_video_list(path=path)] == list("abcde")


def test_save_fetch_result_keeps_cache_on_truncated_fetch(tmp_path):
    """A fetch < 50% of the cached size is treated as likely-incomplete."""
    path = tmp_path / "videos.json"
    cached = [{"video_id": str(i)} for i in range(10)]
    save_video_list(cached, path=path)

    truncated = [{"video_id": "0"}, {"video_id": "1"}]  # 20%
    result = save_fetch_result(truncated, path=path)

    assert len(result) == 10
    assert len(load_video_list(path=path)) == 10


def test_save_fetch_result_saves_plausible_fetch(tmp_path):
    """A plausibly-complete fetch overwrites and is returned."""
    path = tmp_path / "videos.json"
    save_video_list([{"video_id": "a"}, {"video_id": "b"}], path=path)

    fresh = [{"video_id": str(i)} for i in range(5)]
    result = save_fetch_result(fresh, path=path)

    assert [v["video_id"] for v in result] == [str(i) for i in range(5)]
    assert [v["video_id"] for v in load_video_list(path=path)] == [str(i) for i in range(5)]


def test_save_fetch_result_saves_when_no_cache(tmp_path):
    """With no existing cache, even a small/empty fetch is persisted as-is."""
    path = tmp_path / "videos.json"
    fresh = [{"video_id": "only"}]
    result = save_fetch_result(fresh, path=path)
    assert [v["video_id"] for v in result] == ["only"]
    assert [v["video_id"] for v in load_video_list(path=path)] == ["only"]


# ── R1: load_video_list must not swallow real I/O errors ─────────────────────


def test_load_video_list_propagates_oserror(tmp_path, monkeypatch):
    """A genuine read failure on an existing file must NOT be masked as []."""
    path = tmp_path / "videos.json"
    save_video_list([{"video_id": "x"}], path=path)

    def boom(*a, **k):
        raise OSError("disk on fire")

    # Patch Path.read_text so the existing-file read raises a real I/O error.
    monkeypatch.setattr(fetcher.Path, "read_text", boom)
    with pytest.raises(OSError):
        load_video_list(path=path)


# ── R1: save_video_list cleans up the temp file when the write fails ─────────


def test_save_video_list_cleans_temp_on_failure(tmp_path, monkeypatch):
    """If the atomic write fails mid-way, no *.tmp leftover should remain."""
    path = tmp_path / "videos.json"

    def boom(*a, **k):
        raise RuntimeError("fsync failed")

    monkeypatch.setattr(fetcher.os, "fsync", boom)
    with pytest.raises(RuntimeError):
        save_video_list([{"video_id": "x"}], path=path)

    leftovers = [p.name for p in tmp_path.iterdir()]
    assert leftovers == [], f"temp file not cleaned up: {leftovers}"


# ── R1: channel-fetch subprocess timeout is passed and TimeoutExpired skips ──


def test_yt_dlp_fetch_passes_timeout_and_skips_on_timeout(monkeypatch):
    """A hung yt-dlp (TimeoutExpired) is caught per-tab and yields no entries."""
    seen = {}

    def fake_run(cmd, *a, **k):
        seen["timeout"] = k.get("timeout")
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=k.get("timeout"))

    monkeypatch.setattr(fetcher.subprocess, "run", fake_run)
    monkeypatch.setattr(fetcher.shutil, "which", lambda name: "/usr/bin/yt-dlp")

    out = fetcher._fetch_channel_videos_with_yt_dlp(
        "https://youtube.com/c/x", include_streams=False
    )
    assert out == [], "a timed-out fetch must be skipped, not raised"
    assert seen["timeout"] is not None and seen["timeout"] > 0


# ── BUG: flaky approximate_date scrape silently drops the newest videos ──────
#
# yt-dlp's youtubetab:approximate_date extractor intermittently returns a whole
# tab with `timestamp`/`upload_date` missing on EVERY entry. fetch_video_list
# drops undated videos to enforce the age cutoff, so a glitched batch silently
# erases the newest videos — most visibly past livestreams on the /streams tab.
# _scrape_tab_entries retries the per-tab scrape when it detects that
# all-or-nothing dropout.


class _FakeCompleted:
    def __init__(self, stdout):
        self.stdout = stdout


def _payload(entries):
    return json.dumps({"entries": entries})


def _url_aware_fake_run(responses_by_url):
    """Build a fake subprocess.run keyed by a substring of the target URL.

    ``responses_by_url`` maps a url substring to a list of per-attempt entry
    lists; the Nth call for that url returns the Nth list (clamping to the last
    once exhausted). Records call counts so tests can assert retry behaviour.
    """
    state = {"total": 0, "per_url": {}}

    def fake_run(cmd, *a, **k):
        url = cmd[-1]
        state["total"] += 1
        for key, sequence in responses_by_url.items():
            if key in url:
                idx = state["per_url"].get(key, 0)
                state["per_url"][key] = idx + 1
                entries = sequence[min(idx, len(sequence) - 1)]
                return _FakeCompleted(_payload(entries))
        return _FakeCompleted(_payload([]))

    return fake_run, state


def test_scrape_tab_retries_when_whole_batch_is_undated(monkeypatch):
    """A wholesale-undated batch is retried; the dated retry result wins.

    Discriminates the fix: without the retry the first (undated) batch would be
    returned and its entries dropped downstream. We assert the returned entry
    carries the date from the SECOND attempt.
    """
    undated = [{"id": "VID1", "title": "Inside Unreal", "duration": 5600}]
    dated = [
        {
            "id": "VID1",
            "title": "Inside Unreal",
            "duration": 5600,
            "timestamp": 1780617600,
        }
    ]
    fake_run, state = _url_aware_fake_run({"/streams": [undated, dated]})
    monkeypatch.setattr(fetcher.subprocess, "run", fake_run)
    monkeypatch.setattr(fetcher.shutil, "which", lambda name: "/usr/bin/yt-dlp")

    entries = fetcher._scrape_tab_entries(["/usr/bin/yt-dlp"], "https://x/streams")

    assert state["per_url"]["/streams"] == 2, "undated batch must trigger one retry"
    assert len(entries) == 1
    assert entries[0]["timestamp"] == 1780617600, "the dated retry result must win"


def test_scrape_tab_gives_up_and_warns_after_max_attempts(monkeypatch, capsys):
    """If every attempt is undated, the batch is returned (not lost) with a warning.

    The videos are still surfaced to the caller; they will be dropped by the
    age filter (no date to test against), but the operator is told why via
    stderr so the loss is never silent.
    """
    undated = [{"id": "VID1", "title": "Inside Unreal", "duration": 5600}]
    fake_run, state = _url_aware_fake_run({"/streams": [undated]})
    monkeypatch.setattr(fetcher.subprocess, "run", fake_run)
    monkeypatch.setattr(fetcher.shutil, "which", lambda name: "/usr/bin/yt-dlp")

    entries = fetcher._scrape_tab_entries(
        ["/usr/bin/yt-dlp"], "https://x/streams", max_attempts=3
    )

    assert state["per_url"]["/streams"] == 3, "must exhaust all attempts"
    assert len(entries) == 1, "the undated batch is still returned, not swallowed"
    warning = capsys.readouterr().err
    assert "no parseable dates" in warning
    assert "/streams" in warning


def test_scrape_tab_does_not_retry_on_subprocess_error(monkeypatch):
    """A hard subprocess failure/timeout is NOT retried (returns [] at once)."""
    calls = {"n": 0}

    def fake_run(cmd, *a, **k):
        calls["n"] += 1
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=k.get("timeout"))

    monkeypatch.setattr(fetcher.subprocess, "run", fake_run)
    monkeypatch.setattr(fetcher.shutil, "which", lambda name: "/usr/bin/yt-dlp")

    entries = fetcher._scrape_tab_entries(
        ["/usr/bin/yt-dlp"], "https://x/streams", max_attempts=3
    )

    assert entries == []
    assert calls["n"] == 1, "subprocess errors must not be retried"


def test_scrape_tab_no_retry_when_some_entries_dated(monkeypatch):
    """A normal batch where at least one entry is dated is returned immediately.

    Upcoming/scheduled streams legitimately lack a date; their presence
    alongside dated entries must NOT be mistaken for the dropout glitch.
    """
    mixed = [
        {"id": "SCHED", "title": "Upcoming Stream", "duration": None},
        {
            "id": "VID1",
            "title": "Inside Unreal",
            "duration": 5600,
            "timestamp": 1780617600,
        },
    ]
    fake_run, state = _url_aware_fake_run({"/streams": [mixed]})
    monkeypatch.setattr(fetcher.subprocess, "run", fake_run)
    monkeypatch.setattr(fetcher.shutil, "which", lambda name: "/usr/bin/yt-dlp")

    entries = fetcher._scrape_tab_entries(["/usr/bin/yt-dlp"], "https://x/streams")

    assert state["per_url"]["/streams"] == 1, "a partially-dated batch is not retried"
    assert len(entries) == 2


def test_fetch_video_list_recovers_flaky_stream_after_retry(monkeypatch):
    """End-to-end: a new livestream survives a glitched first /streams scrape.

    This is the user-reported scenario — a past livestream (only on /streams)
    vanished from `ingest --update` because the streams scrape came back
    wholesale-undated. With the retry, the second attempt's date lands and the
    video flows all the way through fetch_video_list.
    """
    recent_ts = int(
        (datetime.now(timezone.utc) - timedelta(days=10)).timestamp()
    )
    target_undated = [
        {"id": "STREAM1", "title": "Inside Unreal: Projectiles", "duration": 5600}
    ]
    target_dated = [
        {
            "id": "STREAM1",
            "title": "Inside Unreal: Projectiles",
            "duration": 5600,
            "timestamp": recent_ts,
        }
    ]
    # /videos tab returns nothing; the target lives only on /streams, whose
    # first scrape is the glitched undated batch and second is healthy.
    fake_run, _state = _url_aware_fake_run(
        {"/videos": [[]], "/streams": [target_undated, target_dated]}
    )
    monkeypatch.setattr(fetcher.subprocess, "run", fake_run)
    monkeypatch.setattr(fetcher.shutil, "which", lambda name: "/usr/bin/yt-dlp")

    results = fetch_video_list(include_streams=True)

    ids = {v["video_id"] for v in results}
    assert "STREAM1" in ids, "the new livestream must survive the flaky scrape"


# ── R3 residuals: load_video_list shape validation ──────────────────────────


def test_load_video_list_rejects_non_list_json(tmp_path):
    """Well-formed JSON that isn't a list must return [] (not crash callers)."""
    for payload in ['{"video_id": "x"}', '"hello"', "42", "true"]:
        path = tmp_path / "videos.json"
        path.write_text(payload)
        assert load_video_list(path=path) == [], f"payload {payload!r} should yield []"


def test_load_video_list_drops_non_dict_elements(tmp_path):
    """Non-dict elements in the list are dropped so callers can assume dicts."""
    path = tmp_path / "videos.json"
    path.write_text('[{"video_id": "a"}, "junk", 5, {"video_id": "b"}]')
    loaded = load_video_list(path=path)
    assert [v["video_id"] for v in loaded] == ["a", "b"]


# ── R3 residuals: _parse_relative_time word-boundary safety ─────────────────


def test_parse_relative_time_no_false_positive_on_non_timestamp():
    """Non-timestamp text that merely contains a unit substring returns None."""
    assert _parse_relative_time("a yearly recap") is None
    assert _parse_relative_time("an early second draft") is None
    assert _parse_relative_time("seconds of footage") is None


def test_parse_relative_time_still_parses_genuine_timestamps():
    """Real YouTube relative strings still parse after the word-boundary change."""
    now = datetime.now(timezone.utc)
    assert _parse_relative_time("2 days ago") is not None
    assert _parse_relative_time("Streamed 3 weeks ago") is not None
    a_year = _parse_relative_time("a year ago")
    assert a_year is not None and 360 <= (now - a_year).days <= 370


# ── R3 residuals: save_fetch_result force override ──────────────────────────


def test_save_fetch_result_force_overwrites_shrink(tmp_path):
    """force=True replaces the cache even with a large shrink (intentional)."""
    path = tmp_path / "videos.json"
    save_video_list([{"video_id": str(i)} for i in range(10)], path=path)
    shrunk = [{"video_id": "0"}]  # 10% — would be blocked without force
    result = save_fetch_result(shrunk, path=path, force=True)
    assert [v["video_id"] for v in result] == ["0"]
    assert [v["video_id"] for v in load_video_list(path=path)] == ["0"]
