"""Generate embeddings via the Ollama REST API."""

from __future__ import annotations

import atexit

import requests

from . import config
from .config import (
    EMBEDDING_MODEL,
    EMBEDDING_QUERY_INSTRUCTION,
    EMBEDDING_QUERY_TEMPLATE,
    OLLAMA_BASE_URL,
)


_session = requests.Session()


def _validate_embedding_dim(vectors: list[list[float]]) -> None:
    """Validate returned vectors match the configured embedding dimension.

    Guards against model/config drift: if the Ollama model's output size no
    longer matches config.EMBEDDING_DIM, wrong-sized vectors would otherwise
    be upserted (or fail opaquely deep inside Qdrant). We check cheaply — the
    first vector of the batch is enough to catch a dimension change — and
    raise a clear error pointing at the mismatch.

    config.EMBEDDING_DIM is read dynamically (not imported by value) so tests
    and callers can override it.
    """
    if not vectors:
        return
    expected = config.EMBEDDING_DIM
    got = len(vectors[0])
    if got != expected:
        raise RuntimeError(
            f"Embedding dimension mismatch: model {EMBEDDING_MODEL!r} returned "
            f"vectors of length {got}, but config.EMBEDDING_DIM is {expected}. "
            "The embedding model or EMBEDDING_DIM config has changed; align "
            "them (and recreate the Qdrant collection) before indexing."
        )


def close_session() -> None:
    _session.close()


atexit.register(close_session)


def embed_text(text: str) -> list[float]:
    """Return an embedding vector for a passage of text (document side)."""
    with _session.post(
        f"{OLLAMA_BASE_URL}/api/embed",
        json={"model": EMBEDDING_MODEL, "input": text},
        timeout=300,
    ) as resp:
        resp.raise_for_status()
        data = resp.json()
    return data["embeddings"][0]


def embed_query(query: str, instruction: str | None = None) -> list[float]:
    """Embed a search query for asymmetric retrieval.

    Wraps the query using EMBEDDING_QUERY_TEMPLATE — a format string that
    receives {instruction} and {query}. The default targets Qwen3-Embedding;
    set the env vars to switch templates for other models (e.g. BGE, E5) or
    set EMBEDDING_QUERY_TEMPLATE to "{query}" to disable wrapping entirely.

    Passing `instruction` explicitly overrides EMBEDDING_QUERY_INSTRUCTION
    for this call only.
    """
    instr = instruction if instruction is not None else EMBEDDING_QUERY_INSTRUCTION
    formatted = EMBEDDING_QUERY_TEMPLATE.format(instruction=instr, query=query)
    return embed_text(formatted)



def build_chunk_embed_text(title: str, chunk_text: str) -> str:
    """Compose the text to embed for a transcript chunk.

    Prepending the video title anchors the chunk's embedding to the video's
    actual subject. Without this, a five-second tangential mention of a topic
    in an unrelated video can outrank a chunk from a video whose whole
    subject is that topic but whose speaker uses synonyms.
    """
    return f"{title}\n\n{chunk_text}"


def embed_texts(texts: list[str], batch_size: int = 128) -> list[list[float]]:
    """Embed multiple texts, batching requests to Ollama."""
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        with _session.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={"model": EMBEDDING_MODEL, "input": batch},
            timeout=300,
        ) as resp:
            resp.raise_for_status()
            data = resp.json()
        batch_embeddings = data["embeddings"]
        if len(batch_embeddings) != len(batch):
            raise RuntimeError(
                f"Ollama returned {len(batch_embeddings)} embeddings for a "
                f"batch of {len(batch)} inputs (model {EMBEDDING_MODEL!r}). "
                "Refusing to continue with misaligned embeddings."
            )
        _validate_embedding_dim(batch_embeddings)
        all_embeddings.extend(batch_embeddings)

    if len(all_embeddings) != len(texts):
        raise RuntimeError(
            f"Embedding count mismatch: produced {len(all_embeddings)} "
            f"vectors for {len(texts)} input texts."
        )
    return all_embeddings
