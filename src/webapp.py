"""FastAPI web application."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from sse_starlette.sse import EventSourceResponse

from . import ingest_worker
from .fetcher import load_video_list
from .search import search_videos
from .vectordb import get_client, ensure_collection
from .config import COLLECTION_NAME

app = FastAPI(title="UE Video Search", version="1.0.0")

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


# ── Pages ──────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
async def index():
    return (STATIC_DIR / "index.html").read_text()


# ── Search API ─────────────────────────────────────────


@app.get("/api/search")
async def api_search(
    q: str = Query(..., min_length=1, description="Search query"),
    top_k: int = Query(10, ge=1, le=50),
) -> dict[str, Any]:
    try:
        results = search_videos(q, top_k=top_k)
    except ValueError as e:
        return {"error": str(e), "query": q}
    except ConnectionError as e:
        return {"error": str(e), "query": q}
    except RuntimeError as e:
        return {"error": str(e), "query": q}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}", "query": q}

    # Group by video for the frontend
    grouped: dict[str, dict[str, Any]] = {}
    for r in results:
        vid_url = r["video_url"]
        if vid_url not in grouped:
            grouped[vid_url] = {
                "video_title": r["video_title"],
                "video_url": vid_url,
                "video_id": vid_url.split("v=")[-1] if "v=" in vid_url else "",
                "segments": [],
            }
        grouped[vid_url]["segments"].append(
            {
                "time_range": r["time_range"],
                "timestamped_url": r["timestamped_url"],
                "start_seconds": r["start_seconds"],
                "end_seconds": r["end_seconds"],
                "score": round(r["score"], 4),
                "excerpt": r["excerpt"],
            }
        )

    return {"query": q, "total_results": len(results), "videos": list(grouped.values())}


# ── Stats API ──────────────────────────────────────────


@app.get("/api/stats")
async def api_stats() -> dict[str, Any]:
    try:
        client = get_client()
        info = client.get_collection(COLLECTION_NAME)
        points = info.points_count or 0
    except Exception:
        points = 0

    videos = load_video_list()
    return {
        "indexed_chunks": points,
        "cached_videos": len(videos),
    }


# ── Ingest API ─────────────────────────────────────────


@app.post("/api/ingest")
async def api_ingest_start(
    incremental: bool = Query(False, description="Only process new videos"),
    reindex: bool = Query(False, description="Re-index already indexed videos"),
    skip_uefn: bool = Query(True, description="Skip videos with UEFN or Fortnite in title"),
    skip_automotive: bool = Query(True, description="Skip videos with automotive in title"),
    skip_archvis: bool = Query(True, description="Skip videos with archvis in title"),
) -> dict[str, Any]:
    loop = asyncio.get_running_loop()
    started = ingest_worker.start_ingest(
        loop,
        incremental=incremental,
        reindex=reindex,
        skip_uefn=skip_uefn,
        skip_automotive=skip_automotive,
        skip_archvis=skip_archvis,
    )
    if not started:
        return {"ok": False, "message": "Ingest is already running."}
    return {"ok": True, "message": "Ingest started."}


@app.get("/api/ingest/status")
async def api_ingest_status() -> dict[str, Any]:
    return ingest_worker.get_status()


@app.get("/api/ingest/stream")
async def api_ingest_stream():
    """SSE stream of ingest progress events."""
    queue = ingest_worker.subscribe()

    async def event_generator():
        try:
            # Send current status immediately
            yield {"event": "status", "data": json.dumps(ingest_worker.get_status())}

            while True:
                try:
                    data = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield {"event": "status", "data": json.dumps(data)}
                    if data.get("phase") in ("done", "error"):
                        break
                except asyncio.TimeoutError:
                    # Send keepalive
                    yield {"event": "ping", "data": "{}"}
        finally:
            ingest_worker.unsubscribe(queue)

    return EventSourceResponse(event_generator())



