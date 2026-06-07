"""Tests for src.vectordb.

Unit tests use a stub Qdrant client and never touch a live server.
Integration tests talk to the live Qdrant (env QDRANT_HOST/QDRANT_PORT) and
ALWAYS use a unique throwaway collection that is deleted in teardown. They
refuse to run against the production collection name `ue_videos`, and skip
gracefully when the live server is unreachable.
"""
import pytest

from src import vectordb


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
