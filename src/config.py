"""Central configuration.

All settings can be overridden via environment variables.
"""

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("DATA_DIR", PROJECT_ROOT / "data"))
AUDIO_DIR = DATA_DIR / "audio"
TRANSCRIPT_DIR = DATA_DIR / "transcripts"

# ── YouTube ────────────────────────────────────────────
CHANNEL_URL = os.environ.get("CHANNEL_URL", "https://www.youtube.com/unrealengine")
MAX_AGE_YEARS = int(os.environ.get("MAX_AGE_YEARS", "3"))
MIN_DURATION_SECONDS = int(os.environ.get("MIN_DURATION_SECONDS", str(15 * 60)))

# ── Whisper ────────────────────────────────────────────
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")

# ── Ollama ─────────────────────────────────────────────
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "qwen3-embedding:0.6b")
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "1024"))

# Asymmetric retrieval: many embedding models (Qwen3, BGE, E5, ...) expect
# queries to be wrapped differently from documents. The template receives
# {instruction} and {query}; reference whichever placeholders the model needs.
# Set the template to "{query}" to disable wrapping entirely.
EMBEDDING_QUERY_INSTRUCTION = os.environ.get(
    "EMBEDDING_QUERY_INSTRUCTION",
    "Given a search query about Unreal Engine, retrieve transcript passages "
    "from technical videos that answer the query.",
)
EMBEDDING_QUERY_TEMPLATE = os.environ.get(
    "EMBEDDING_QUERY_TEMPLATE",
    "Instruct: {instruction}\nQuery: {query}",
)

# ── Qdrant ─────────────────────────────────────────────
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "ue_videos")

# ── Chunking ──────────────────────────────────────────
CHUNK_DURATION_SECONDS = int(os.environ.get("CHUNK_DURATION_SECONDS", "120"))
CHUNK_OVERLAP_SECONDS = int(os.environ.get("CHUNK_OVERLAP_SECONDS", "15"))
