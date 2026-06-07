"""Shared pytest fixtures. Tests run without live Qdrant/Ollama/GPU."""
import sys
from pathlib import Path

# Ensure repo root is importable as `src`
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
