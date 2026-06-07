"""Tests for the FastAPI web application (src/webapp.py).

These tests run without a live Ollama/Qdrant/GPU backend by monkeypatching
the blocking dependencies that webapp delegates to.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

import src.webapp as webapp


# ── Bug 1: blocking calls offloaded to a thread executor ──────────────


def _sample_results() -> list[dict[str, Any]]:
    """Two segments across two videos, in the shape api_search expects."""
    return [
        {
            "video_url": "https://www.youtube.com/watch?v=AAA",
            "video_title": "First Video",
            "time_range": "00:00 - 00:30",
            "timestamped_url": "https://www.youtube.com/watch?v=AAA&t=0s",
            "start_seconds": 0.0,
            "end_seconds": 30.0,
            "score": 0.987654,
            "excerpt": "hello world",
        },
        {
            "video_url": "https://www.youtube.com/watch?v=AAA",
            "video_title": "First Video",
            "time_range": "00:30 - 01:00",
            "timestamped_url": "https://www.youtube.com/watch?v=AAA&t=30s",
            "start_seconds": 30.0,
            "end_seconds": 60.0,
            "score": 0.5,
            "excerpt": "more text",
        },
        {
            "video_url": "https://www.youtube.com/watch?v=BBB",
            "video_title": "Second Video",
            "time_range": "00:00 - 00:15",
            "timestamped_url": "https://www.youtube.com/watch?v=BBB&t=0s",
            "start_seconds": 0.0,
            "end_seconds": 15.0,
            "score": 0.25,
            "excerpt": "another excerpt",
        },
    ]


def test_api_search_groups_results(monkeypatch):
    captured: dict[str, Any] = {}

    def fake_search(q, top_k=10):
        captured["q"] = q
        captured["top_k"] = top_k
        return _sample_results()

    monkeypatch.setattr(webapp, "search_videos", fake_search)

    with TestClient(webapp.app) as client:
        resp = client.get("/api/search", params={"q": "test", "top_k": 5})

    assert resp.status_code == 200
    body = resp.json()

    # The executor path must forward the query + top_k unchanged.
    assert captured == {"q": "test", "top_k": 5}

    assert body["query"] == "test"
    assert body["total_results"] == 3
    assert len(body["videos"]) == 2

    by_url = {v["video_url"]: v for v in body["videos"]}
    first = by_url["https://www.youtube.com/watch?v=AAA"]
    assert first["video_title"] == "First Video"
    assert first["video_id"] == "AAA"
    assert len(first["segments"]) == 2
    # score is rounded to 4 dp in the response.
    assert first["segments"][0]["score"] == 0.9877
    assert first["segments"][0]["time_range"] == "00:00 - 00:30"
    assert first["segments"][0]["excerpt"] == "hello world"

    second = by_url["https://www.youtube.com/watch?v=BBB"]
    assert second["video_id"] == "BBB"
    assert len(second["segments"]) == 1


def test_api_search_value_error(monkeypatch):
    def boom(q, top_k=10):
        raise ValueError("query is empty")

    monkeypatch.setattr(webapp, "search_videos", boom)

    with TestClient(webapp.app) as client:
        resp = client.get("/api/search", params={"q": "x"})

    assert resp.status_code == 200
    body = resp.json()
    assert body == {"error": "query is empty", "query": "x"}


def test_api_search_unexpected_error(monkeypatch):
    def boom(q, top_k=10):
        raise KeyError("kaboom")

    monkeypatch.setattr(webapp, "search_videos", boom)

    with TestClient(webapp.app) as client:
        resp = client.get("/api/search", params={"q": "y"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["query"] == "y"
    assert body["error"].startswith("An unexpected error occurred:")


def test_api_stats_shape(monkeypatch):
    class FakeInfo:
        points_count = 42

    class FakeClient:
        def get_collection(self, name):
            return FakeInfo()

    monkeypatch.setattr(webapp, "get_client", lambda: FakeClient())
    monkeypatch.setattr(webapp, "load_video_list", lambda: [{"video_id": "a"}, {"video_id": "b"}])

    with TestClient(webapp.app) as client:
        resp = client.get("/api/stats")

    assert resp.status_code == 200
    assert resp.json() == {"indexed_chunks": 42, "cached_videos": 2}


def test_api_stats_handles_backend_error(monkeypatch):
    def broken_client():
        raise ConnectionError("qdrant down")

    monkeypatch.setattr(webapp, "get_client", broken_client)
    monkeypatch.setattr(webapp, "load_video_list", lambda: [{"video_id": "a"}])

    with TestClient(webapp.app) as client:
        resp = client.get("/api/stats")

    assert resp.status_code == 200
    assert resp.json() == {"indexed_chunks": 0, "cached_videos": 1}


# ── Bug 2: SSE stream terminates for clients connecting post-ingest ───
#
# These drive the SSE async generator directly (deterministic, no network)
# under asyncio.wait_for so a regression (the old code blocked on queue.get()
# forever, emitting 30s keepalives) FAILS the test promptly instead of
# hanging the whole suite for 30s+.


def _terminal_status(phase: str = "done") -> dict[str, Any]:
    return {
        "phase": phase,
        "total": 1,
        "completed": 1,
        "skipped": 0,
        "failed": 0,
        "current_video": "",
        "message": "Ingest complete." if phase == "done" else "Ingest error: boom",
        "new_videos_found": 0,
    }


async def _drain_generator(gen, *, item_timeout: float = 2.0) -> list[dict[str, Any]]:
    """Consume an async generator, bounding each step so a hang surfaces as
    a TimeoutError rather than blocking forever."""
    out: list[dict[str, Any]] = []
    try:
        while True:
            item = await asyncio.wait_for(gen.__anext__(), timeout=item_timeout)
            out.append(item)
    except StopAsyncIteration:
        return out
    finally:
        await gen.aclose()


@pytest.mark.parametrize("phase", ["done", "error"])
def test_ingest_stream_terminates_when_already_terminal(monkeypatch, phase):
    """A client connecting after ingest finished must NOT block forever."""
    monkeypatch.setattr(webapp.ingest_worker, "get_status", lambda: _terminal_status(phase))

    # Track that the subscriber queue is always cleaned up (no leak).
    unsub_calls: list[Any] = []
    real_unsubscribe = webapp.ingest_worker.unsubscribe

    def tracking_unsubscribe(q):
        unsub_calls.append(q)
        return real_unsubscribe(q)

    monkeypatch.setattr(webapp.ingest_worker, "unsubscribe", tracking_unsubscribe)

    queue = webapp.ingest_worker.subscribe()
    gen = webapp._ingest_event_generator(queue)

    # asyncio.wait_for around the whole drain: a regression that loops on
    # keepalives would blow this 5s budget and fail the test (not hang).
    events = asyncio.run(asyncio.wait_for(_drain_generator(gen), timeout=5.0))

    # Exactly one event: the terminal status snapshot. No keepalive pings.
    assert len(events) == 1
    assert events[0]["event"] == "status"
    payload = json.loads(events[0]["data"])
    assert payload["phase"] == phase
    # finally block ran -> queue unsubscribed (no leak).
    assert unsub_calls == [queue]


def test_ingest_stream_live_path_then_terminal(monkeypatch):
    """Sanity: the non-terminal live path still streams queued updates and
    stops on a terminal item received from the queue (refactor preserved it)."""
    monkeypatch.setattr(webapp.ingest_worker, "get_status", lambda: _terminal_status("processing"))

    queue = webapp.ingest_worker.subscribe()
    # Pre-load a live update followed by a terminal update.
    queue.put_nowait({"phase": "processing", "message": "working"})
    queue.put_nowait(_terminal_status("done"))

    gen = webapp._ingest_event_generator(queue)
    events = asyncio.run(asyncio.wait_for(_drain_generator(gen), timeout=5.0))

    phases = [json.loads(e["data"])["phase"] for e in events]
    # snapshot(processing) -> live(processing) -> live(done) then stop.
    assert phases == ["processing", "processing", "done"]
    assert all(e["event"] == "status" for e in events)
    webapp.ingest_worker.unsubscribe(queue)


# ── Bug 1 regression guard: work is actually offloaded off the event loop ──


def _spy_run_in_executor(monkeypatch):
    """Wrap the running loop's run_in_executor and record the callable threads.

    Returns a dict that, after a request, holds:
      - "offloaded": True if run_in_executor was actually used, and
      - "loop_thread"/"work_thread": the thread the endpoint coroutine ran on
        vs the thread the blocking callable ran on (they MUST differ).
    A reverted direct-blocking call never touches run_in_executor, so
    "offloaded" stays False and the assertions fail — making this discriminate
    the fix (unlike comparing against the main thread, which the TestClient
    portal thread defeats).
    """
    import asyncio.base_events
    import threading

    info: dict[str, Any] = {"offloaded": False, "work_threads": [], "loop_threads": []}
    # run_in_executor is defined on BaseEventLoop (the concrete loop's class),
    # not the abstract base — patch there so the wrapper is actually hit.
    base = asyncio.base_events.BaseEventLoop
    orig = base.run_in_executor

    def wrapper(self, executor, func, *args):
        info["offloaded"] = True
        info["loop_threads"].append(threading.current_thread())

        def wrapped():
            info["work_threads"].append(threading.current_thread())
            return func(*args)

        return orig(self, executor, wrapped)

    monkeypatch.setattr(base, "run_in_executor", wrapper)
    return info


def test_api_search_offloads_to_executor_thread(monkeypatch):
    """search_videos must be dispatched via run_in_executor, on a different
    thread than the endpoint coroutine. Reverting to a direct call fails this."""
    monkeypatch.setattr(webapp, "search_videos", lambda q, top_k=10: [])
    info = _spy_run_in_executor(monkeypatch)

    client = TestClient(webapp.app)
    resp = client.get("/api/search", params={"q": "x"})
    assert resp.status_code == 200
    assert info["offloaded"], "search_videos was not dispatched via run_in_executor"
    assert info["work_threads"], "executor callable never ran"
    # The blocking work ran on a thread distinct from the coroutine's loop thread.
    assert set(info["work_threads"]).isdisjoint(set(info["loop_threads"])), (
        f"work ran on the loop thread: work={info['work_threads']} loop={info['loop_threads']}"
    )


def test_api_stats_offloads_to_executor_thread(monkeypatch):
    """The blocking Qdrant/file work in api_stats must go through run_in_executor."""
    class FakeClient:
        def get_collection(self, name):
            class _Info:
                points_count = 7

            return _Info()

    monkeypatch.setattr(webapp, "get_client", lambda: FakeClient())
    monkeypatch.setattr(webapp, "load_video_list", lambda *a, **k: [{"video_id": "a"}])
    info = _spy_run_in_executor(monkeypatch)

    client = TestClient(webapp.app)
    resp = client.get("/api/stats")
    assert resp.status_code == 200
    assert info["offloaded"], "api_stats did not offload blocking work to run_in_executor"
    assert set(info["work_threads"]).isdisjoint(set(info["loop_threads"])), (
        f"stats work ran on the loop thread: work={info['work_threads']} loop={info['loop_threads']}"
    )
