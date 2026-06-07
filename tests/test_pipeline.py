"""Hermetic tests for src.pipeline.

No real network, Qdrant, or Whisper. Heavy collaborators are monkeypatched.
"""
from __future__ import annotations

import src.pipeline as p


def _vid(video_id: str) -> dict:
    return {"video_id": video_id, "title": f"title {video_id}", "url": f"http://x/{video_id}"}


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
