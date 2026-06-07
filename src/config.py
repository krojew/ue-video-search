"""Central configuration.

All settings can be overridden via environment variables.
"""

import os
from pathlib import Path


class ConfigError(ValueError):
    """Raised when an environment-derived setting is invalid."""


def _env_int(name, default, *, min_value=None, max_value=None):
    """Read an int setting from the environment with clear error reporting.

    Parses ``os.environ[name]`` (falling back to ``default``) as an int and
    applies optional inclusive bounds. On any failure raises a ``ConfigError``
    naming the variable, its offending value, and the violated constraint.
    """
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except (TypeError, ValueError):
        # `from None` suppresses the chained int() ValueError so the operator
        # sees only the clear, actionable ConfigError message.
        raise ConfigError(
            f"{name} must be an integer, got {raw!r}."
        ) from None
    if min_value is not None and value < min_value:
        raise ConfigError(
            f"{name} must be >= {min_value}, got {value}."
        )
    if max_value is not None and value > max_value:
        raise ConfigError(
            f"{name} must be <= {max_value}, got {value}."
        )
    return value


# ── Paths ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("DATA_DIR", PROJECT_ROOT / "data"))
AUDIO_DIR = DATA_DIR / "audio"
TRANSCRIPT_DIR = DATA_DIR / "transcripts"

# ── YouTube ────────────────────────────────────────────
CHANNEL_URL = os.environ.get("CHANNEL_URL", "https://www.youtube.com/unrealengine")
MAX_AGE_YEARS = _env_int("MAX_AGE_YEARS", 3, min_value=1)
MIN_DURATION_SECONDS = _env_int("MIN_DURATION_SECONDS", 15 * 60, min_value=0)

# ── Whisper ────────────────────────────────────────────
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")

# ── Ollama ─────────────────────────────────────────────
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "qwen3-embedding:0.6b")
EMBEDDING_DIM = _env_int("EMBEDDING_DIM", 1024, min_value=1)

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
QDRANT_PORT = _env_int("QDRANT_PORT", 6333, min_value=1, max_value=65535)
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "ue_videos")

# ── Chunking ──────────────────────────────────────────
CHUNK_DURATION_SECONDS = _env_int("CHUNK_DURATION_SECONDS", 120, min_value=1)
CHUNK_OVERLAP_SECONDS = _env_int("CHUNK_OVERLAP_SECONDS", 15, min_value=0)
if CHUNK_OVERLAP_SECONDS >= CHUNK_DURATION_SECONDS:
    raise ConfigError(
        f"CHUNK_OVERLAP_SECONDS ({CHUNK_OVERLAP_SECONDS}) must be less than "
        f"CHUNK_DURATION_SECONDS ({CHUNK_DURATION_SECONDS})."
    )
