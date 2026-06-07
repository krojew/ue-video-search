"""Tests for src.vectordb.

Unit tests use a stub Qdrant client and never touch a live server.
Integration tests talk to the live Qdrant (env QDRANT_HOST/QDRANT_PORT) and
ALWAYS use a unique throwaway collection that is deleted in teardown. They
refuse to run against the production collection name `ue_videos`, and skip
gracefully when the live server is unreachable.
"""
import os
import random
from uuid import uuid4

import pytest

from src import vectordb

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    VectorParams,
)


# ── Live-Qdrant integration support ───────────────────────────────────────

QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
LIVE_EMBEDDING_DIM = 1024


def _live_client_or_skip() -> QdrantClient:
    """Connect to the live Qdrant or skip the test if unreachable."""
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=5)
        client.get_collections()  # forces a real round-trip
    except Exception as exc:  # pragma: no cover - depends on environment
        pytest.skip(f"Live Qdrant unreachable at {QDRANT_HOST}:{QDRANT_PORT}: {exc}")
    return client


@pytest.fixture
def live_collection(monkeypatch):
    """Yield (client, collection_name) for a unique throwaway collection.

    Hard-guards against the production collection name and tears the
    throwaway collection down afterwards. Patches vectordb's module-level
    COLLECTION_NAME and EMBEDDING_DIM so the functions under test operate on
    the throwaway collection with the live embedding dimension.
    """
    client = _live_client_or_skip()
    name = f"test_{uuid4().hex}"
    # Safety: never operate against the real production collection.
    assert name != "ue_videos"
    assert os.environ.get("COLLECTION_NAME") != "ue_videos" or name != "ue_videos"

    monkeypatch.setattr(vectordb, "COLLECTION_NAME", name)
    monkeypatch.setattr(vectordb, "EMBEDDING_DIM", LIVE_EMBEDDING_DIM)
    try:
        yield client, name
    finally:
        try:
            client.delete_collection(collection_name=name)
        except Exception:
            pass
        client.close()


class _StubClient:
    """A Qdrant client stub that records calls but talks to nothing."""

    def __init__(self):
        self.upserted = []

    def get_collections(self):
        class _C:
            collections = []

        return _C()

    def create_collection(self, *args, **kwargs):
        pass

    def upsert(self, *args, **kwargs):
        self.upserted.append((args, kwargs))


# ── BUG 1: upsert_chunks must reject mismatched chunk/embedding counts ─────


def test_upsert_chunks_raises_on_length_mismatch():
    stub = _StubClient()
    chunks = [
        {"start": 0, "end": 10, "text": "a"},
        {"start": 10, "end": 20, "text": "b"},
    ]
    embeddings = [[0.1] * 4]  # only one vector for two chunks
    with pytest.raises(ValueError):
        vectordb.upsert_chunks(
            "vid", "title", "url", chunks, embeddings, client=stub
        )
    # Nothing should have been upserted.
    assert stub.upserted == []


# ── BUG 2 (integration): dim mismatch on existing collection must raise ────


def test_ensure_collection_raises_on_dim_mismatch(live_collection):
    client, name = live_collection
    # Create the collection with the WRONG size (8 != EMBEDDING_DIM 1024).
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=8, distance=Distance.COSINE),
    )
    with pytest.raises(RuntimeError):
        vectordb.ensure_collection(client)


def test_ensure_collection_ok_when_dim_matches(live_collection):
    client, name = live_collection
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(
            size=LIVE_EMBEDDING_DIM, distance=Distance.COSINE
        ),
    )
    # Matching dimension: must not raise.
    vectordb.ensure_collection(client)


# ── BUG 3 (integration): re-ingest must replace, not orphan, old chunks ────


def _rand_vec(dim=LIVE_EMBEDDING_DIM):
    return [random.random() for _ in range(dim)]


def _count_for_video(client, name, video_id):
    return client.count(
        collection_name=name,
        count_filter=Filter(
            must=[
                FieldCondition(key="video_id", match=MatchValue(value=video_id))
            ]
        ),
        exact=True,
    ).count


def test_reindex_replaces_old_chunks(live_collection):
    client, name = live_collection
    vid = "vtest"

    # First ingest: chunks at starts [0, 120].
    chunks_v1 = [
        {"start": 0, "end": 120, "text": "first"},
        {"start": 120, "end": 240, "text": "second"},
    ]
    emb_v1 = [_rand_vec(), _rand_vec()]
    vectordb.upsert_chunks(vid, "title", "url", chunks_v1, emb_v1, client=client)
    assert _count_for_video(client, name, vid) == 2

    # Re-ingest with DIFFERENT windowing: starts [0, 100, 200].
    chunks_v2 = [
        {"start": 0, "end": 100, "text": "a"},
        {"start": 100, "end": 200, "text": "b"},
        {"start": 200, "end": 300, "text": "c"},
    ]
    emb_v2 = [_rand_vec(), _rand_vec(), _rand_vec()]
    vectordb.upsert_chunks(vid, "title", "url", chunks_v2, emb_v2, client=client)

    # Exactly 3 points: the old start=120 orphan must be gone.
    assert _count_for_video(client, name, vid) == 3

    # And no point with start=120 should remain.
    points, _ = client.scroll(
        collection_name=name,
        scroll_filter=Filter(
            must=[FieldCondition(key="video_id", match=MatchValue(value=vid))]
        ),
        limit=100,
        with_payload=True,
    )
    starts = sorted((p.payload or {}).get("start") for p in points)
    assert starts == [0, 100, 200]


def test_reindex_does_not_touch_other_videos(live_collection):
    client, name = live_collection
    # Two distinct videos; re-ingesting one must not delete the other's points.
    vectordb.upsert_chunks(
        "vid_a", "ta", "ua",
        [{"start": 0, "end": 10, "text": "a"}],
        [_rand_vec()],
        client=client,
    )
    vectordb.upsert_chunks(
        "vid_b", "tb", "ub",
        [{"start": 0, "end": 10, "text": "b"}],
        [_rand_vec()],
        client=client,
    )
    # Re-ingest vid_a with new windowing.
    vectordb.upsert_chunks(
        "vid_a", "ta", "ua",
        [{"start": 5, "end": 15, "text": "a2"}],
        [_rand_vec()],
        client=client,
    )
    assert _count_for_video(client, name, "vid_a") == 1
    assert _count_for_video(client, name, "vid_b") == 1


# ── Hermetic: _existing_vector_size handles named-vectors and None ──────────


class _StubVectors:
    """Stand-in for the .config.params.vectors attribute shape."""
    def __init__(self, vectors):
        self._v = vectors

    @property
    def config(self):
        return self

    @property
    def params(self):
        return self

    @property
    def vectors(self):
        return self._v


class _StubVectorClient:
    def __init__(self, vectors):
        self._vectors = vectors

    def get_collection(self, name):
        return _StubVectors(self._vectors)


def test_existing_vector_size_unnamed():
    client = _StubVectorClient(VectorParams(size=1024, distance=Distance.COSINE))
    assert vectordb._existing_vector_size(client, "c") == 1024


def test_existing_vector_size_named_single():
    client = _StubVectorClient({"text": VectorParams(size=768, distance=Distance.COSINE)})
    assert vectordb._existing_vector_size(client, "c") == 768


def test_existing_vector_size_named_conflicting_returns_none():
    client = _StubVectorClient({
        "a": VectorParams(size=768, distance=Distance.COSINE),
        "b": VectorParams(size=1024, distance=Distance.COSINE),
    })
    assert vectordb._existing_vector_size(client, "c") is None
