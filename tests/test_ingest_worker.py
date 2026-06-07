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


def test_bug3_concurrent_start_ingest_starts_exactly_one(monkeypatch):
    """Many concurrent callers must still yield exactly one started ingest.

    NOTE: this is a best-effort stress test, not a strict lock discriminator.
    The check-and-set window in start_ingest is tiny, so an unlocked version
    may still pass here most of the time; it is kept as a smoke test for the
    common concurrent path. The atomic reservation under _lock is the real
    guard, and the constructor/launch rollback tests below pin the related
    invariants deterministically.
    """
    n = 32
    started: list[object] = []
    started_lock = threading.Lock()
    # Capture the REAL Thread before patching — the callers below must use it,
    # since the patch replaces threading.Thread that start_ingest constructs.
    RealThread = threading.Thread

    class FakeThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            with started_lock:
                started.append(self)

    monkeypatch.setattr(w.threading, "Thread", FakeThread)
    monkeypatch.setattr(w, "_run_ingest", lambda *a, **k: None)

    barrier = threading.Barrier(n)
    results: list[bool] = []
    results_lock = threading.Lock()

    def caller():
        barrier.wait()  # maximise contention on the check-and-set
        r = w.start_ingest(object(), incremental=True)
        with results_lock:
            results.append(r)

    threads = [RealThread(target=caller) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert results.count(True) == 1, f"exactly one caller must win, got {results.count(True)}"
    assert len(started) == 1, f"exactly one worker thread may start, got {len(started)}"


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


def test_bug3_thread_construction_failure_rolls_back_running(monkeypatch):
    """If Thread.__init__ (not start) raises, _running must still roll back.

    Regression guard for the HIGH review finding: the Thread(...) construction
    must be inside the rollback try, or a constructor failure bricks ingest by
    leaving _running stuck True.
    """

    class ExplodingConstructor:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("cannot allocate thread")

    monkeypatch.setattr(w.threading, "Thread", ExplodingConstructor)
    monkeypatch.setattr(w, "_run_ingest", lambda *a, **k: None)

    with pytest.raises(RuntimeError):
        w.start_ingest(object(), incremental=True)

    assert w.is_running() is False, "_running must roll back on constructor failure"


_EXPECTED_STATUS_KEYS = {
    "phase",
    "total",
    "completed",
    "skipped",
    "failed",
    "current_video",
    "message",
    "new_videos_found",
}


def test_bug4_update_status_is_atomic_and_complete(monkeypatch):
    """_update_status applies fields/increments under lock and emits a full dict."""
    w._status = w.IngestStatus()
    w._update_status(total=3, message="hi")
    snap = w.get_status()
    assert snap["total"] == 3
    assert snap["message"] == "hi"
    assert set(snap.keys()) == _EXPECTED_STATUS_KEYS

    w._update_status(inc={"completed": 1, "failed": 1})
    w._update_status(inc={"completed": 1})
    snap = w.get_status()
    assert snap["completed"] == 2
    assert snap["failed"] == 1


def test_bug4_concurrent_get_status_never_raises(monkeypatch):
    """Hammering get_status() while _run_ingest mutates _status must not tear."""
    videos = [_vid(str(i)) for i in range(8)]
    monkeypatch.setattr(w, "load_video_list", lambda: list(videos))
    monkeypatch.setattr(w, "fetch_video_list", lambda **k: list(videos))
    monkeypatch.setattr(w, "save_video_list", lambda v: None)
    monkeypatch.setattr(w, "save_fetch_result", lambda v: list(v))

    processed: list[str] = []
    _patch_heavy(monkeypatch, indexed_ids=set(), processed=processed)

    errors: list[BaseException] = []
    stop = threading.Event()

    def reader():
        while not stop.is_set():
            try:
                snap = w.get_status()
                assert set(snap.keys()) == _EXPECTED_STATUS_KEYS
                # counters must always be plain ints, never partially updated
                assert isinstance(snap["completed"], int)
                assert isinstance(snap["failed"], int)
            except BaseException as exc:  # noqa: BLE001 - record any tear/exception
                errors.append(exc)
                return

    readers = [threading.Thread(target=reader) for _ in range(4)]
    for r in readers:
        r.start()

    w._run_ingest(incremental=False, reindex=False)

    stop.set()
    for r in readers:
        r.join(timeout=5)

    assert not errors, f"concurrent get_status raised/tore: {errors[:1]}"
    final = w.get_status()
    assert final["phase"] == w.IngestPhase.DONE.value


def test_bug4_lock_prevents_torn_read_of_coupled_fields(monkeypatch):
    """Lock-sensitive guard: a reader must NEVER see a half-applied update.

    _update_status sets multiple fields under _lock. We install a status whose
    setattr sleeps *between* the two coupled writes (completed then failed),
    deliberately widening the window. With the lock, get_status() serialises
    and always observes completed == failed. Without the lock (the unfixed
    code) a reader interleaves mid-update and sees completed != failed. This
    test FAILS if _lock is removed — unlike the smoke test above.
    """
    import time

    class SlowStatus(w.IngestStatus):
        # Re-declare as a dataclass-free subclass: it inherits fields and
        # to_dict; we only override attribute writes to inject a delay.
        def __setattr__(self, name, value):
            if name == "failed":
                time.sleep(0.001)  # widen the torn-read window
            object.__setattr__(self, name, value)

    w._status = SlowStatus()

    seen_torn: list[tuple[int, int]] = []
    stop = threading.Event()

    def reader():
        while not stop.is_set():
            snap = w.get_status()
            if snap["completed"] != snap["failed"]:
                seen_torn.append((snap["completed"], snap["failed"]))

    readers = [threading.Thread(target=reader) for _ in range(4)]
    for r in readers:
        r.start()

    # Writer keeps the invariant completed == failed across each atomic update.
    for n in range(1, 60):
        w._update_status(completed=n, failed=n)
        time.sleep(0.0005)

    stop.set()
    for r in readers:
        r.join(timeout=5)

    assert not seen_torn, (
        "reader observed a torn (completed != failed) snapshot — _lock is not "
        f"protecting coupled updates: {seen_torn[:3]}"
    )


def test_bug5_malformed_video_does_not_abort_run(monkeypatch):
    """A malformed entry is counted failed; the run completes and indexes the good one."""
    good = _vid("good")
    malformed = {"not_a_video": True}  # missing video_id/title/url
    videos = [malformed, good]

    monkeypatch.setattr(w, "load_video_list", lambda: list(videos))
    monkeypatch.setattr(w, "fetch_video_list", lambda **k: list(videos))
    monkeypatch.setattr(w, "save_video_list", lambda v: None)
    monkeypatch.setattr(w, "save_fetch_result", lambda v: list(v))

    processed: list[str] = []
    # reindex=True -> skip_indexed off -> index diff bypassed, so the malformed
    # entry reaches the per-video loop (the BUG5 code path).
    _patch_heavy(monkeypatch, indexed_ids=set(), processed=processed)

    w._run_ingest(incremental=False, reindex=True)

    snap = w.get_status()
    assert snap["phase"] == w.IngestPhase.DONE.value, "run must complete, not error out"
    assert processed == ["good"], "only the well-formed video should be processed"
    assert snap["failed"] == 1, "the malformed entry must be counted as failed"
    assert snap["completed"] == 2, "both entries advance the loop (1 failed + 1 ok)"


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
