"""Search interface: embed a query and find matching video segments."""

from __future__ import annotations

import re
from typing import Any

from .embeddings import embed_text
from .vectordb import search as vector_search


def _extract_keywords(text: str) -> set[str]:
    """Extract significant keywords from text (lowercased, minimum 3 chars)."""
    # Remove common words
    stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "from", "with", "by"}

    # Extract words (alphanumeric and hyphens)
    words = re.findall(r'\b\w+\b', text.lower())

    # Filter: >= 3 chars and not stop words
    keywords = {w for w in words if len(w) >= 3 and w not in stop_words}
    return keywords


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
        # Fetch more results than requested to allow for title-based reranking
        raw_results = vector_search(query_embedding, top_k=top_k * 2)
    except Exception as e:
        if "Connection refused" in str(e) or "timeout" in str(e).lower():
            raise ConnectionError("Unable to connect to vector database. Please ensure Qdrant is running.") from e
        if "collection" in str(e).lower() and "not found" in str(e).lower():
            raise RuntimeError("No video data has been indexed yet. Please run an ingest first.") from e
        raise RuntimeError(f"Search failed: {e}") from e

    # Extract keywords from query for title matching
    query_keywords = _extract_keywords(query)
    title_keywords_lower = {kw.lower() for kw in query_keywords}

    # Boost scores for results with matching video titles
    for result in raw_results:
        title_lower = result["video_title"].lower()
        # Count how many query keywords appear in the title
        title_matches = sum(1 for kw in title_keywords_lower if kw in title_lower)

        if title_matches > 0:
            # Boost score proportionally to matches (0.1 points per keyword match)
            result["score"] = result["score"] + (0.1 * title_matches)

    # Cap scores at 1.0 to prevent them from exceeding 100%
    for result in raw_results:
        result["score"] = min(result["score"], 1.0)

    # Sort by boosted score and take top_k
    raw_results.sort(key=lambda r: r["score"], reverse=True)
    raw_results = raw_results[:top_k]

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
