"""Tests for the FastAPI web application (src/webapp.py).

These tests run without a live Ollama/Qdrant/GPU backend by monkeypatching
the blocking dependencies that webapp delegates to.
"""
from __future__ import annotations

import asyncio
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
