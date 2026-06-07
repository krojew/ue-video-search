"""Hermetic tests for src.pipeline.

No real network, Qdrant, or Whisper. Heavy collaborators are monkeypatched.
"""
from __future__ import annotations

import src.pipeline as p


def _vid(video_id: str) -> dict:
    return {"video_id": video_id, "title": f"title {video_id}", "url": f"http://x/{video_id}"}


def test_bug2_run_fetch_delegates_to_save_fetch_result(monkeypatch):
    """run_fetch must persist via the shared guarded helper, not raw save.

    The empty/truncated-fetch guard lives in fetcher.save_fetch_result (tested
    directly in test_fetcher.py). Here we verify run_fetch routes the fresh
    fetch through that helper and returns whatever it decides is authoritative.
    """
    fresh = [_vid(str(i)) for i in range(5)]
    monkeypatch.setattr(p, "fetch_video_list", lambda **k: list(fresh))

    calls = []

    def fake_save_fetch_result(videos):
        calls.append(list(videos))
        return list(videos)

    monkeypatch.setattr(p, "save_fetch_result", fake_save_fetch_result)
    # run_fetch must NOT call raw save_video_list directly any more.
    monkeypatch.setattr(
        p, "save_video_list",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("raw save bypassed guard")),
    )

    result = p.run_fetch(use_cached=False)

    assert len(calls) == 1, "run_fetch must delegate to save_fetch_result exactly once"
    assert [v["video_id"] for v in calls[0]] == [str(i) for i in range(5)]
    assert [v["video_id"] for v in result] == [str(i) for i in range(5)]


def test_bug2_run_fetch_returns_retained_cache_when_guard_keeps_it(monkeypatch):
    """When the guard keeps the cache, run_fetch returns that (not the fresh)."""
    fresh: list[dict] = []  # transient empty fetch
    retained = [_vid(c) for c in "abcde"]
    monkeypatch.setattr(p, "fetch_video_list", lambda **k: list(fresh))
    # Simulate the guard deciding to keep the existing cache.
    monkeypatch.setattr(p, "save_fetch_result", lambda videos: list(retained))

    result = p.run_fetch(use_cached=False)

    assert [v["video_id"] for v in result] == list("abcde")


def test_bug2_use_cached_true_unchanged(monkeypatch):
    """use_cached=True must return the cache without ever fetching/saving."""
    cached = [_vid("a")]
    monkeypatch.setattr(p, "load_video_list", lambda: list(cached))

    def boom(**k):
        raise AssertionError("fetch_video_list must not be called when use_cached=True")

    monkeypatch.setattr(p, "fetch_video_list", boom)
    saved = []
    monkeypatch.setattr(p, "save_fetch_result", lambda videos: saved.append(list(videos)))

    result = p.run_fetch(use_cached=True)

    assert [v["video_id"] for v in result] == ["a"]
    assert saved == []


def test_bug5_malformed_video_does_not_abort_ingest(monkeypatch):
    """A malformed entry must be skipped, the good one still indexed."""
    good = _vid("good")
    malformed = {"not_a_video": True}  # missing video_id/title/url
    to_process = [malformed, good]

    processed: list[str] = []

    def fake_process_video(vid, url, model=None):
        processed.append(vid)
        return [{"text": "hi", "start": 0.0, "end": 1.0}]

    upserts: list[str] = []

    monkeypatch.setattr(p, "get_client", lambda: object())
    monkeypatch.setattr(p, "ensure_collection", lambda client: None)
    monkeypatch.setattr(p, "load_whisper_model", lambda: object())
    monkeypatch.setattr(p, "process_video", fake_process_video)
    monkeypatch.setattr(p, "build_chunk_embed_text", lambda title, text: text)
    monkeypatch.setattr(p, "embed_texts", lambda texts: [[0.0] for _ in texts])
    monkeypatch.setattr(
        p, "upsert_chunks", lambda vid, *a, **k: upserts.append(vid) or 1
    )
    monkeypatch.setattr(p, "load_transcript", lambda video_id: None)
    monkeypatch.setattr(p, "download_audio", lambda video_id, url: None)

    class _FakeCuda:
        @staticmethod
        def is_available():
            return False

        @staticmethod
        def empty_cache():
            return None

    monkeypatch.setattr(p.torch, "cuda", _FakeCuda)

    # skip_indexed=False bypasses the index diff so the malformed entry reaches
    # the per-video loop (the code path BUG5 hardens). Must not raise.
    p._ingest_videos(to_process, skip_indexed=False, label="t")

    assert processed == ["good"], "only the well-formed video should be processed"
    assert upserts == ["good"], "only the good video should be upserted"


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
