"""Generate embeddings via the Ollama REST API."""

from __future__ import annotations

import atexit

import requests

from .config import EMBEDDING_MODEL, OLLAMA_BASE_URL


_session = requests.Session()


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
    """Embed a search query with the Qwen3 instruction template.

    Qwen3-Embedding is asymmetric: queries must be prefixed with a task
    instruction so they land in the same region of the space as documents.
    """
    if instruction is None:
        instruction = (
            "Given a search query about Unreal Engine, retrieve transcript "
            "passages from technical videos that answer the query."
        )
    formatted = f"Instruct: {instruction}\nQuery: {query}"
    return embed_text(formatted)


def embed_texts(texts: list[str], batch_size: int = 32) -> list[list[float]]:
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
        all_embeddings.extend(data["embeddings"])
    return all_embeddings
