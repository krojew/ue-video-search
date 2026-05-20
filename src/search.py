"""Search interface: embed a query and find matching video segments."""

from __future__ import annotations

import re
from typing import Any

from .embeddings import embed_query
from .vectordb import search as vector_search


_TITLE_BOOST_PER_KEYWORD = 0.1
_MAX_RESULTS_PER_VIDEO = 3
_ADJACENT_GAP_SECONDS = 30.0


def _group_adjacent_chunks(results: list[dict[str, Any]], gap: float) -> list[dict[str, Any]]:
    """Merge same-video chunks that overlap or sit within `gap` seconds.

    The merged result keeps the highest-scoring chunk's text and start time
    (so the timestamp link lands on the most relevant moment) but extends the
    end time to cover the union of the merged range.
    """
    by_video: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        by_video.setdefault(r["video_id"], []).append(r)

    merged: list[dict[str, Any]] = []
    for items in by_video.values():
        items.sort(key=lambda r: r["start"])
        current: dict[str, Any] | None = None
        for r in items:
            if current is None:
                current = dict(r)
                continue
            if r["start"] <= current["end"] + gap:
                current["end"] = max(current["end"], r["end"])
                if r["score"] > current["score"]:
                    current["score"] = r["score"]
                    current["text"] = r["text"]
                    current["start"] = r["start"]
            else:
                merged.append(current)
                current = dict(r)
        if current is not None:
            merged.append(current)
    return merged


def _cap_per_video(results: list[dict[str, Any]], max_per_video: int) -> list[dict[str, Any]]:
    """Keep at most `max_per_video` highest-scoring results per video."""
    seen: dict[str, int] = {}
    out: list[dict[str, Any]] = []
    for r in sorted(results, key=lambda x: x["score"], reverse=True):
        vid = r["video_id"]
        if seen.get(vid, 0) >= max_per_video:
            continue
        seen[vid] = seen.get(vid, 0) + 1
        out.append(r)
    return out


_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "nor", "so", "yet", "as", "if",
    "in", "on", "at", "to", "for", "of", "from", "with", "by", "about",
    "into", "onto", "upon", "over", "under", "between", "through",
    "is", "are", "was", "were", "be", "been", "being", "am",
    "have", "has", "had", "having",
    "do", "does", "did", "doing", "done",
    "will", "would", "should", "could", "can", "may", "might", "must", "shall",
    "i", "me", "my", "you", "your", "he", "him", "his", "she", "her", "it", "its",
    "we", "us", "our", "they", "them", "their",
    "this", "that", "these", "those",
    "what", "which", "who", "whom", "when", "where", "why", "how",
    "all", "any", "some", "no", "not", "only", "same", "than", "too", "very",
    "just", "also", "more", "most", "much", "many", "few",
    "use", "using", "used", "make", "making", "made",
    "get", "got", "getting", "let", "lets",
    "tutorial", "tutorials", "guide", "intro", "video", "part",
})

_TECH_ACRONYMS: frozenset[str] = frozenset({
    "ue", "ue4", "ue5", "ai", "ar", "vr", "xr", "gi", "ml",
    "ui", "ux", "ik", "fk", "io", "os", "fov", "fps",
    "lod", "hlod", "pcg", "bsp", "hud", "gpu", "cpu",
    "sdf", "rtx", "api", "sdk", "vfx", "sfx",
    "rgb", "hdr", "png", "jpg", "fbx", "tga",
})


def _extract_keywords(text: str) -> set[str]:
    """Extract significant keywords from text for title-boost matching.

    - Allows internal hyphens so "real-time" / "post-process" stay intact.
    - Drops a wide list of high-frequency English words that would otherwise
      cause every "How to ..." tutorial title to match every query.
    - Allows short UE-relevant acronyms (UE5, AI, AR, VR, ML, GI, IK, FK,
      LOD, PCG, GPU, etc.) that the >=3-char filter would otherwise drop.
    """
    tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())
    keywords: set[str] = set()
    for tok in tokens:
        if tok in _STOP_WORDS:
            continue
        if len(tok) >= 3 or tok in _TECH_ACRONYMS:
            keywords.add(tok)
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
        query_embedding = embed_query(query)
    except Exception as e:
        if "Connection refused" in str(e) or "timeout" in str(e).lower():
            raise ConnectionError("Unable to connect to embedding service. Please ensure Ollama is running and the embedding model is available.") from e
        raise RuntimeError(f"Failed to generate embeddings for query: {e}") from e

    try:
        # Over-fetch so grouping/diversification have raw material to work with.
        raw_results = vector_search(query_embedding, top_k=top_k * 4)
    except Exception as e:
        if "Connection refused" in str(e) or "timeout" in str(e).lower():
            raise ConnectionError("Unable to connect to vector database. Please ensure Qdrant is running.") from e
        if "collection" in str(e).lower() and "not found" in str(e).lower():
            raise RuntimeError("No video data has been indexed yet. Please run an ingest first.") from e
        raise RuntimeError(f"Search failed: {e}") from e

    title_keywords_lower = {kw.lower() for kw in _extract_keywords(query)}
    for result in raw_results:
        title_lower = result["video_title"].lower()
        title_matches = sum(1 for kw in title_keywords_lower if kw in title_lower)
        if title_matches > 0:
            result["score"] = result["score"] + (_TITLE_BOOST_PER_KEYWORD * title_matches)

    grouped = _group_adjacent_chunks(raw_results, _ADJACENT_GAP_SECONDS)
    diversified = _cap_per_video(grouped, _MAX_RESULTS_PER_VIDEO)

    diversified.sort(key=lambda r: r["score"], reverse=True)
    diversified = diversified[:top_k]

    results = []
    for r in diversified:
        results.append(
            {
                "video_title": r["video_title"],
                "video_url": r["video_url"],
                "timestamped_url": _youtube_timestamp_url(r["video_url"], r["start"]),
                "time_range": f"{_seconds_to_hms(r['start'])} → {_seconds_to_hms(r['end'])}",
                "start_seconds": r["start"],
                "end_seconds": r["end"],
                "score": min(r["score"], 1.0),
                "excerpt": r["text"][:500],
            }
        )

    return results
