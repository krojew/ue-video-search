"""Hermetic tests for src.pipeline.

No real network, Qdrant, or Whisper. Heavy collaborators are monkeypatched.
"""
from __future__ import annotations

import src.pipeline as p


def _vid(video_id: str) -> dict:
    return {"video_id": video_id, "title": f"title {video_id}", "url": f"http://x/{video_id}"}


def test_bug2_empty_fetch_does_not_overwrite_cache(monkeypatch):
    """A transient empty fetch must not clobber a populated cache."""
    cached = [_vid(c) for c in "abcde"]
    monkeypatch.setattr(p, "load_video_list", lambda: list(cached))
    monkeypatch.setattr(p, "fetch_video_list", lambda **k: [])

    saved = []
    monkeypatch.setattr(p, "save_video_list", lambda videos: saved.append(list(videos)))

    result = p.run_fetch(use_cached=False)

    assert saved == [], "save_video_list must not be called on empty fetch"
    ids = [v["video_id"] for v in result]
    assert ids == list("abcde"), "must return the existing cached list"


def test_bug2_truncated_fetch_does_not_overwrite_cache(monkeypatch):
    """A fetch far smaller than the cache (< 50%) must not overwrite it."""
    cached = [_vid(str(i)) for i in range(10)]
    truncated = [_vid("0"), _vid("1")]  # only 20% of the cache
    monkeypatch.setattr(p, "load_video_list", lambda: list(cached))
    monkeypatch.setattr(p, "fetch_video_list", lambda **k: list(truncated))

    saved = []
    monkeypatch.setattr(p, "save_video_list", lambda videos: saved.append(list(videos)))

    result = p.run_fetch(use_cached=False)

    assert saved == [], "save_video_list must not overwrite with truncated fetch"
    assert len(result) == 10, "must keep the full cached list"


def test_bug2_valid_fetch_still_saves(monkeypatch):
    """A plausibly-complete fetch still overwrites the cache as before."""
    cached = [_vid("a"), _vid("b")]
    fresh = [_vid(str(i)) for i in range(5)]
    monkeypatch.setattr(p, "load_video_list", lambda: list(cached))
    monkeypatch.setattr(p, "fetch_video_list", lambda **k: list(fresh))

    saved = []
    monkeypatch.setattr(p, "save_video_list", lambda videos: saved.append(list(videos)))

    result = p.run_fetch(use_cached=False)

    assert len(saved) == 1, "valid fetch must be persisted"
    assert [v["video_id"] for v in saved[0]] == [str(i) for i in range(5)]
    assert len(result) == 5


def test_bug2_use_cached_true_unchanged(monkeypatch):
    """use_cached=True must return the cache without ever fetching/saving."""
    cached = [_vid("a")]
    monkeypatch.setattr(p, "load_video_list", lambda: list(cached))

    def boom(**k):
        raise AssertionError("fetch_video_list must not be called when use_cached=True")

    monkeypatch.setattr(p, "fetch_video_list", boom)
    saved = []
    monkeypatch.setattr(p, "save_video_list", lambda videos: saved.append(list(videos)))

    result = p.run_fetch(use_cached=True)

    assert [v["video_id"] for v in result] == ["a"]
    assert saved == []


def test_bug1_run_ingest_new_only_processes_full_merged_list(monkeypatch):
    """Incremental CLI ingest must hand the *merged* list to _ingest_videos.

    Even with zero brand-new videos, cached-but-unindexed videos must remain
    eligible. _ingest_videos(skip_indexed=True) does the index diff, so the
    selection union is achieved by passing the whole merged list.
    """
    merged = [_vid("a"), _vid("b"), _vid("c")]
    new_only: list[dict] = []  # nothing brand-new this run

    monkeypatch.setattr(
        p, "run_fetch_incremental", lambda **k: (list(merged), list(new_only))
    )

    captured = {}

    def fake_ingest(videos, skip_indexed=True, label="Processing videos"):
        captured["videos"] = videos
        captured["skip_indexed"] = skip_indexed

    monkeypatch.setattr(p, "_ingest_videos", fake_ingest)

    p.run_ingest_new_only()

    assert captured["skip_indexed"] is True
    ids = [v["video_id"] for v in captured["videos"]]
    assert ids == ["a", "b", "c"], "must pass the full merged list, not just new_only"
