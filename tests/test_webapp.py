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


def test_api_search_offloads_to_executor_thread(monkeypatch):
    """search_videos must run on an executor worker, not the event-loop thread.

    Reverting `await loop.run_in_executor(None, ...)` back to a direct blocking
    call would make this fail (the callable would run on the main/loop thread).
    """
    import threading

    main_thread = threading.main_thread()
    seen: dict[str, Any] = {}

    def fake_search(q, top_k=10):
        seen["thread"] = threading.current_thread()
        return []

    monkeypatch.setattr(webapp, "search_videos", fake_search)

    client = TestClient(webapp.app)
    resp = client.get("/api/search", params={"q": "x"})
    assert resp.status_code == 200
    assert "thread" in seen, "search_videos was never called"
    assert seen["thread"] is not main_thread, (
        "search_videos ran on the event-loop/main thread — it was not offloaded "
        "to run_in_executor, so it would block the loop"
    )


def test_api_stats_offloads_to_executor_thread(monkeypatch):
    """The blocking Qdrant/file work in api_stats must run off the loop thread."""
    import threading

    main_thread = threading.main_thread()
    seen: list[str] = []

    class FakeClient:
        def get_collection(self, name):
            seen.append(f"get_collection:{threading.current_thread() is main_thread}")

            class _Info:
                points_count = 7

            return _Info()

    monkeypatch.setattr(webapp, "get_client", lambda: FakeClient())

    def fake_load_video_list(*a, **k):
        seen.append(f"load:{threading.current_thread() is main_thread}")
        return [{"video_id": "a"}]

    monkeypatch.setattr(webapp, "load_video_list", fake_load_video_list)

    client = TestClient(webapp.app)
    resp = client.get("/api/stats")
    assert resp.status_code == 200
    # Both blocking calls must report running off the main thread (False).
    assert seen, "stats backend was never called"
    assert all(s.endswith("False") for s in seen), f"stats work ran on loop thread: {seen}"
