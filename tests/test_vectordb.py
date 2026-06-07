"""Tests for src.vectordb.

Unit tests use a stub Qdrant client and never touch a live server.
Integration tests talk to the live Qdrant (env QDRANT_HOST/QDRANT_PORT) and
ALWAYS use a unique throwaway collection that is deleted in teardown. They
refuse to run against the production collection name `ue_videos`, and skip
gracefully when the live server is unreachable.
"""
import os
from uuid import uuid4

import pytest

from src import vectordb

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


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
