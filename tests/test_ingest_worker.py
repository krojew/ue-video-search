"""Hermetic tests for src.ingest_worker.

No real network, Qdrant, Whisper, or asyncio event loop is used; every heavy
collaborator is monkeypatched. Tests reset module-global state between runs.
"""
from __future__ import annotations

import threading

import pytest

import src.ingest_worker as w


@pytest.fixture(autouse=True)
def _reset_worker_state():
    """Reset module-global ingest state before and after each test."""
    w._running = False
    w._status = w.IngestStatus()
    w._event_queues = []
    w._event_loop = None
    yield
    w._running = False
    w._status = w.IngestStatus()
    w._event_queues = []
    w._event_loop = None


def _vid(video_id: str) -> dict:
    return {"video_id": video_id, "title": f"title {video_id}", "url": f"http://x/{video_id}"}


def _patch_heavy(monkeypatch, *, indexed_ids: set[str], processed: list[str]):
    """Patch every collaborator past the selection logic so _run_ingest is hermetic.

    `processed` is appended with each video_id that reaches process_video.
    """
    # No event loop / queues -> _emit is effectively a no-op, but keep it cheap.
    monkeypatch.setattr(w, "get_client", lambda: object())
    monkeypatch.setattr(w, "ensure_collection", lambda client: None)
    monkeypatch.setattr(w, "list_indexed_video_ids", lambda client: set(indexed_ids))
    monkeypatch.setattr(w, "load_whisper_model", lambda: object())
    monkeypatch.setattr(w, "build_chunk_embed_text", lambda title, text: text)
    monkeypatch.setattr(w, "embed_texts", lambda texts: [[0.0] for _ in texts])
    monkeypatch.setattr(w, "upsert_chunks", lambda *a, **k: 1)
    # Prefetch path: never touch the network.
    monkeypatch.setattr(w, "load_transcript", lambda video_id: None)
    monkeypatch.setattr(w, "download_audio", lambda video_id, url: None)

    def fake_process_video(video_id, url, model=None):
        processed.append(video_id)
        return [{"text": "hello", "start": 0.0, "end": 1.0}]

    monkeypatch.setattr(w, "process_video", fake_process_video)

    # Avoid touching CUDA.
    class _FakeCuda:
        @staticmethod
        def is_available():
            return False

        @staticmethod
        def empty_cache():
            return None

    monkeypatch.setattr(w.torch, "cuda", _FakeCuda)


def test_bug1_incremental_retries_cached_video_missing_from_index(monkeypatch):
    """A previously-cached video absent from Qdrant must be re-processed.

    The cached list already knows about "old_failed" (so merge yields no new
    videos), but it never made it into the index. Incremental selection must
    therefore include it: process = new ∪ (cached not in index).
    """
    cached = [_vid("old_indexed"), _vid("old_failed")]
    fresh = [_vid("old_indexed"), _vid("old_failed")]  # nothing brand-new

    monkeypatch.setattr(w, "load_video_list", lambda: list(cached))
    monkeypatch.setattr(w, "fetch_video_list", lambda **k: list(fresh))
    saved = []
    monkeypatch.setattr(w, "save_video_list", lambda videos: saved.append(list(videos)))
    # Use the real merge so new_only is genuinely empty here.

    processed: list[str] = []
    # Index already has old_indexed but is MISSING old_failed.
    _patch_heavy(monkeypatch, indexed_ids={"old_indexed"}, processed=processed)

    w._run_ingest(incremental=True, reindex=False)

    assert "old_failed" in processed, "cached-but-unindexed video must be retried"
    assert "old_indexed" not in processed, "already-indexed video must be skipped"
    # Merged cache is still persisted (BUG1 keeps saving the LIST).
    assert saved, "merged video list should still be saved"


def test_bug3_start_ingest_is_not_double_started(monkeypatch):
    """Two rapid start_ingest calls must start exactly one worker thread."""
    started = []

    class FakeThread:
        def __init__(self, *args, **kwargs):
            self._kwargs = kwargs

        def start(self):
            started.append(self)

    monkeypatch.setattr(w.threading, "Thread", FakeThread)
    # _run_ingest must never actually run (so _running stays reserved).
    monkeypatch.setattr(w, "_run_ingest", lambda *a, **k: None)

    loop = object()  # never touched because the fake thread no-ops
    first = w.start_ingest(loop, incremental=True)
    second = w.start_ingest(loop, incremental=True)

    assert first is True
    assert second is False
    assert len(started) == 1, "only one ingest thread may be started"
    assert w.is_running() is True


def test_bug3_thread_start_failure_rolls_back_running(monkeypatch):
    """If the thread fails to launch, _running must not stay stuck True."""

    class ExplodingThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            raise RuntimeError("cannot start thread")

    monkeypatch.setattr(w.threading, "Thread", ExplodingThread)
    monkeypatch.setattr(w, "_run_ingest", lambda *a, **k: None)

    with pytest.raises(RuntimeError):
        w.start_ingest(object(), incremental=True)

    assert w.is_running() is False, "_running must be rolled back on launch failure"


def test_bug1_incremental_reindex_off_unchanged_when_all_indexed(monkeypatch):
    """When everything cached is already indexed, nothing is processed."""
    cached = [_vid("a"), _vid("b")]
    monkeypatch.setattr(w, "load_video_list", lambda: list(cached))
    monkeypatch.setattr(w, "fetch_video_list", lambda **k: list(cached))
    monkeypatch.setattr(w, "save_video_list", lambda videos: None)

    processed: list[str] = []
    _patch_heavy(monkeypatch, indexed_ids={"a", "b"}, processed=processed)

    w._run_ingest(incremental=True, reindex=False)

    assert processed == []
    assert w._status.phase == w.IngestPhase.DONE
