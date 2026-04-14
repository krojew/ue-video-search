"""Search interface: embed a query and find matching video segments."""

from __future__ import annotations

from typing import Any

from .embeddings import embed_text
from .vectordb import search as vector_search


def _seconds_to_hms(seconds: float) -> str:
    """Convert seconds to H:MM:SS format."""
    h = int(seconds) // 3600
    m = (int(seconds) % 3600) // 60
    s = int(seconds) % 60
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _youtube_timestamp_url(video_url: str, start_seconds: float) -> str:
    """Append a timestamp parameter to a YouTube URL."""
    t = int(start_seconds)
    sep = "&" if "?" in video_url else "?"
    return f"{video_url}{sep}t={t}s"


def search_videos(query: str, top_k: int = 10) -> list[dict[str, Any]]:
    """Search indexed videos for a topic.

    Returns a list of results, each containing:
      - video_title, video_url, timestamped_url
      - start/end times (human-readable)
      - relevance score
      - matching transcript excerpt

    Raises:
      - ValueError: If query is empty or invalid
      - ConnectionError: If embedding service is unavailable
      - RuntimeError: If vector database is unavailable or no collection exists
    """
    if not query or not query.strip():
        raise ValueError("Search query cannot be empty")

    try:
        query_embedding = embed_text(query)
    except Exception as e:
        if "Connection refused" in str(e) or "timeout" in str(e).lower():
            raise ConnectionError("Unable to connect to embedding service. Please ensure Ollama is running and the embedding model is available.") from e
        raise RuntimeError(f"Failed to generate embeddings for query: {e}") from e

    try:
        raw_results = vector_search(query_embedding, top_k=top_k)
    except Exception as e:
        if "Connection refused" in str(e) or "timeout" in str(e).lower():
            raise ConnectionError("Unable to connect to vector database. Please ensure Qdrant is running.") from e
        if "collection" in str(e).lower() and "not found" in str(e).lower():
            raise RuntimeError("No video data has been indexed yet. Please run an ingest first.") from e
        raise RuntimeError(f"Search failed: {e}") from e

    results = []
    for r in raw_results:
        results.append(
            {
                "video_title": r["video_title"],
                "video_url": r["video_url"],
                "timestamped_url": _youtube_timestamp_url(r["video_url"], r["start"]),
                "time_range": f"{_seconds_to_hms(r['start'])} → {_seconds_to_hms(r['end'])}",
                "start_seconds": r["start"],
                "end_seconds": r["end"],
                "score": r["score"],
                "excerpt": r["text"][:500],
            }
        )

    # Deduplicate: if multiple chunks from the same video are adjacent, group them
    return results
