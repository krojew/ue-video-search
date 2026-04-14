"""Generate embeddings via the Ollama REST API."""

from __future__ import annotations

import requests

from .config import EMBEDDING_MODEL, OLLAMA_BASE_URL


def embed_text(text: str) -> list[float]:
    """Return an embedding vector for a single text string."""
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/embed",
        json={"model": EMBEDDING_MODEL, "input": text},
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    # Ollama returns {"embeddings": [[...]]} for /api/embed
    return data["embeddings"][0]


def embed_texts(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Embed multiple texts, batching requests to Ollama."""
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={"model": EMBEDDING_MODEL, "input": batch},
            timeout=300,
        )
        resp.raise_for_status()
        data = resp.json()
        all_embeddings.extend(data["embeddings"])
    return all_embeddings
