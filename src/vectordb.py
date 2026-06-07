"""Qdrant vector database operations."""

from __future__ import annotations

import atexit
import threading
import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    HasIdCondition,
    MatchAny,
    MatchValue,
    PointStruct,
    VectorParams,
)

from .config import COLLECTION_NAME, EMBEDDING_DIM, QDRANT_HOST, QDRANT_PORT


_client: QdrantClient | None = None
_client_lock = threading.Lock()


def get_client() -> QdrantClient:
    global _client
    # Double-checked locking: the lazy singleton is reachable from multiple
    # threads (the ingest worker thread and webapp executor workers), so an
    # unsynchronized check-then-create could build — and leak — two clients.
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=300)
    return _client


def close_client() -> None:
    global _client
    if _client is None:
        return
    client = _client
    close = getattr(client, "close", None)
    if callable(close):
        close()
    _client = None


atexit.register(close_client)


def _existing_vector_size(client: QdrantClient, name: str) -> int | None:
    """Return the configured vector size of an existing collection.

    Handles both the unnamed/default vector config (a VectorParams with a
    `.size`) and the named-vectors form (a dict of name -> VectorParams).
    Returns None if the size cannot be determined.
    """
    params = client.get_collection(name).config.params.vectors
    size = getattr(params, "size", None)
    if size is not None:
        return size
    if isinstance(params, dict):
        sizes = {getattr(vp, "size", None) for vp in params.values()}
        sizes.discard(None)
        if len(sizes) == 1:
            return next(iter(sizes))
    return None


def ensure_collection(client: QdrantClient | None = None) -> None:
    """Create the collection if it doesn't exist.

    If the collection already exists, validate that its vector size matches
    EMBEDDING_DIM. A mismatch means the embedding model / EMBEDDING_DIM
    config drifted away from what the collection was created with; later
    upserts and searches would then fail cryptically (or silently degrade),
    so we raise a clear RuntimeError instructing the operator to recreate
    the collection.
    """
    client = client or get_client()
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in collections:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE,
            ),
        )
        return

    existing_size = _existing_vector_size(client, COLLECTION_NAME)
    if existing_size is not None and existing_size != EMBEDDING_DIM:
        raise RuntimeError(
            f"Collection {COLLECTION_NAME!r} has vector size {existing_size}, "
            f"but EMBEDDING_DIM is {EMBEDDING_DIM}. The embedding model or "
            "EMBEDDING_DIM config has changed; the collection must be "
            "recreated (drop and re-index) to match the new dimension."
        )


def upsert_chunks(
    video_id: str,
    video_title: str,
    video_url: str,
    chunks: list[dict[str, Any]],
    embeddings: list[list[float]],
    client: QdrantClient | None = None,
) -> int:
    """Insert chunk embeddings with metadata into Qdrant. Returns count inserted."""
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"chunks/embeddings length mismatch for video {video_id!r}: "
            f"{len(chunks)} chunks vs {len(embeddings)} embeddings. "
            "Refusing to upsert misaligned data (would silently drop chunks)."
        )

    client = client or get_client()
    ensure_collection(client)

    points = []
    new_ids: list[str] = []
    for chunk, vector in zip(chunks, embeddings):
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{video_id}:{chunk['start']}"))
        new_ids.append(point_id)
        points.append(
            PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "video_id": video_id,
                    "video_title": video_title,
                    "video_url": video_url,
                    "start": chunk["start"],
                    "end": chunk["end"],
                    "text": chunk["text"],
                },
            )
        )

    # Upsert the new points FIRST (in batches), then delete only this video's
    # stale points — those whose id is not among the ones we just wrote.
    # Ordering matters: if we deleted first and the upsert then failed
    # transiently, the video would be left with ZERO points (unsearchable).
    # Upserting first means a mid-operation failure leaves the previous points
    # intact, and re-ingest is still idempotent because identical windowing
    # reproduces the same deterministic ids (overwrite-by-id). The trailing
    # delete drops orphans left behind when the chunk windowing changes.
    batch_size = 100
    for i in range(0, len(points), batch_size):
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points[i : i + batch_size],
        )

    must: list[Any] = [FieldCondition(key="video_id", match=MatchValue(value=video_id))]
    must_not: list[Any] = []
    if new_ids:
        must_not.append(HasIdCondition(has_id=new_ids))
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=FilterSelector(filter=Filter(must=must, must_not=must_not)),
    )

    return len(points)


def video_already_indexed(video_id: str, client: QdrantClient | None = None) -> bool:
    """Check if a video has already been indexed in Qdrant."""
    client = client or get_client()
    try:
        result = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(
                must=[FieldCondition(key="video_id", match=MatchValue(value=video_id))]
            ),
            limit=1,
        )
        return len(result[0]) > 0
    except Exception:
        return False



def list_indexed_video_ids(client: QdrantClient | None = None) -> set[str]:
    """Return the set of distinct video_ids currently present in the collection."""
    client = client or get_client()
    try:
        collections = [c.name for c in client.get_collections().collections]
    except Exception:
        return set()
    if COLLECTION_NAME not in collections:
        return set()

    ids: set[str] = set()
    offset: Any = None
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=None,
            limit=10_000,
            with_payload=["video_id"],
            with_vectors=False,
            offset=offset,
        )
        for p in points:
            vid = (p.payload or {}).get("video_id")
            if vid:
                ids.add(vid)
        if offset is None:
            break
    return ids


def purge_videos_outside(
    allowed_ids: set[str],
    client: QdrantClient | None = None,
) -> tuple[int, int]:
    """Delete points whose video_id is not in `allowed_ids`.

    Returns (videos_purged, points_purged). A safety guard refuses to purge
    when allowed_ids is empty — that would wipe the whole collection, which
    is almost never what a stale-cleanup caller intends.
    """
    if not allowed_ids:
        return (0, 0)

    client = client or get_client()
    indexed_ids = list_indexed_video_ids(client)
    stale_ids = indexed_ids - allowed_ids
    if not stale_ids:
        return (0, 0)

    stale_filter = Filter(
        must=[FieldCondition(key="video_id", match=MatchAny(any=list(stale_ids)))]
    )
    points_purged = client.count(
        collection_name=COLLECTION_NAME,
        count_filter=stale_filter,
        exact=True,
    ).count

    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=FilterSelector(filter=stale_filter),
    )

    return (len(stale_ids), points_purged)


def search(
    query_vector: list[float],
    top_k: int = 10,
    client: QdrantClient | None = None,
) -> list[dict[str, Any]]:
    """Search for the most similar chunks. Returns list of result dicts."""
    client = client or get_client()

    hits = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )

    results = []
    for hit in hits.points:
        payload = hit.payload or {}
        results.append(
            {
                "score": hit.score,
                "video_id": payload.get("video_id", ""),
                "video_title": payload.get("video_title", ""),
                "video_url": payload.get("video_url", ""),
                "start": payload.get("start", 0),
                "end": payload.get("end", 0),
                "text": payload.get("text", ""),
            }
        )
    return results
