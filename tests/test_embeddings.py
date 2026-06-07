"""Tests for src.embeddings.

Unit tests are hermetic: they monkeypatch the module-level requests Session
so no live Ollama is needed.
"""
import pytest

from src import embeddings


class _FakeResponse:
    """Minimal stand-in for a requests.Response used as a context manager."""

    def __init__(self, payload):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _fake_post_factory(embeddings_payload):
    """Return a fake _session.post that ignores args and returns a fixed payload."""

    def _fake_post(*args, **kwargs):
        return _FakeResponse({"embeddings": embeddings_payload})

    return _fake_post


# ── BUG 1: count mismatch must raise, matching counts must align ──────────


def test_embed_texts_raises_when_ollama_returns_fewer(monkeypatch):
    texts = ["a", "b", "c"]
    # Ollama returns only 2 vectors for 3 inputs -> must raise.
    monkeypatch.setattr(
        embeddings._session,
        "post",
        _fake_post_factory([[0.1] * 4, [0.2] * 4]),
    )
    with pytest.raises(RuntimeError):
        embeddings.embed_texts(texts)


def test_embed_texts_aligned_when_counts_match(monkeypatch):
    monkeypatch.setattr(embeddings.config, "EMBEDDING_DIM", 4)
    texts = ["a", "b", "c"]
    expected = [[0.1] * 4, [0.2] * 4, [0.3] * 4]
    monkeypatch.setattr(
        embeddings._session,
        "post",
        _fake_post_factory(expected),
    )
    result = embeddings.embed_texts(texts)
    assert result == expected
    assert len(result) == len(texts)


def test_embed_texts_batches_each_validated(monkeypatch):
    # Two batches of size 2; second batch comes back short -> raise on COUNT
    # (vectors are the right dimension, so this isolates the count check).
    monkeypatch.setattr(embeddings.config, "EMBEDDING_DIM", 4)
    calls = {"n": 0}

    def _fake_post(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return _FakeResponse({"embeddings": [[0.1] * 4, [0.2] * 4]})
        # second batch should have 2 vectors but returns 1
        return _FakeResponse({"embeddings": [[0.3] * 4]})

    monkeypatch.setattr(embeddings._session, "post", _fake_post)
    with pytest.raises(RuntimeError):
        embeddings.embed_texts(["a", "b", "c", "d"], batch_size=2)


# ── BUG 2: wrong embedding dimension must raise ───────────────────────────


def test_embed_texts_raises_on_wrong_dimension(monkeypatch):
    # Counts match (1 vector for 1 input) but the vector is the wrong length.
    monkeypatch.setattr(embeddings.config, "EMBEDDING_DIM", 1024)
    monkeypatch.setattr(
        embeddings._session,
        "post",
        _fake_post_factory([[0.1] * 8]),  # 8 != 1024
    )
    with pytest.raises(RuntimeError):
        embeddings.embed_texts(["only one"])


def test_embed_texts_accepts_correct_dimension(monkeypatch):
    monkeypatch.setattr(embeddings.config, "EMBEDDING_DIM", 4)
    expected = [[0.1] * 4, [0.2] * 4]
    monkeypatch.setattr(
        embeddings._session,
        "post",
        _fake_post_factory(expected),
    )
    assert embeddings.embed_texts(["a", "b"]) == expected
