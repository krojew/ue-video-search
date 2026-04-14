"""Qdrant vector database operations."""

from __future__ import annotations

import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from .config import COLLECTION_NAME, EMBEDDING_DIM, QDRANT_HOST, QDRANT_PORT


def get_client() -> QdrantClient:
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def ensure_collection(client: QdrantClient | None = None) -> None:
    """Create the collection if it doesn't exist."""
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


def upsert_chunks(
    video_id: str,
    video_title: str,
    video_url: str,
    chunks: list[dict[str, Any]],
    embeddings: list[list[float]],
    client: QdrantClient | None = None,
) -> int:
    """Insert chunk embeddings with metadata into Qdrant. Returns count inserted."""
    client = client or get_client()
    ensure_collection(client)

    points = []
    for chunk, vector in zip(chunks, embeddings):
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{video_id}:{chunk['start']}"))
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

    # Upsert in batches of 100
    batch_size = 100
    for i in range(0, len(points), batch_size):
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points[i : i + batch_size],
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
