"""Tests for main.py CLI robustness (interactive crash, exit codes, top-k bounds).

The search backend is monkeypatched so no network/Qdrant/Ollama is needed.
Commands import `search_videos` inside the function body via
`from src.search import search_videos`, so we patch `src.search.search_videos`.
"""
import src.search as search_mod
from click.testing import CliRunner

import main as cli_mod


# --- BUG 2: real errors must produce a non-zero exit code -------------------

def test_search_connection_error_exits_nonzero(monkeypatch):
    def boom(query, top_k=10):
        raise ConnectionError("Ollama is down")

    monkeypatch.setattr(search_mod, "search_videos", boom)

    runner = CliRunner()
    result = runner.invoke(cli_mod.cli, ["search", "foo"])

    assert result.exit_code != 0
    # Friendly message, not a raw traceback.
    assert "Connection Error" in result.output
    assert "Traceback" not in result.output


def test_search_runtime_error_exits_nonzero(monkeypatch):
    def boom(query, top_k=10):
        raise RuntimeError("Qdrant exploded")

    monkeypatch.setattr(search_mod, "search_videos", boom)

    runner = CliRunner()
    result = runner.invoke(cli_mod.cli, ["search", "foo"])

    assert result.exit_code != 0
    assert "Search Error" in result.output
    assert "Traceback" not in result.output


def test_search_value_error_exits_nonzero(monkeypatch):
    def boom(query, top_k=10):
        raise ValueError("bad query")

    monkeypatch.setattr(search_mod, "search_videos", boom)

    runner = CliRunner()
    result = runner.invoke(cli_mod.cli, ["search", "foo"])

    assert result.exit_code != 0
    # Pin the ValueError branch specifically: its message must surface, and it
    # must NOT have fallen through to the generic 'Unexpected Error' handler.
    assert "bad query" in result.output
    assert "Unexpected Error" not in result.output
    assert "Traceback" not in result.output


def test_search_unexpected_error_exits_nonzero(monkeypatch):
    def boom(query, top_k=10):
        raise KeyError("surprise")

    monkeypatch.setattr(search_mod, "search_videos", boom)

    runner = CliRunner()
    result = runner.invoke(cli_mod.cli, ["search", "foo"])

    assert result.exit_code != 0
    assert "Unexpected Error" in result.output
    assert "Traceback" not in result.output


# --- BUG 2 happy path: "no results" is NOT an error (exit 0) ----------------

def test_search_no_results_exits_zero(monkeypatch):
    monkeypatch.setattr(search_mod, "search_videos", lambda query, top_k=10: [])

    runner = CliRunner()
    result = runner.invoke(cli_mod.cli, ["search", "foo"])

    assert result.exit_code == 0
    assert "No results found" in result.output


def test_search_with_results_exits_zero(monkeypatch):
    fake = [
        {
            "video_url": "https://youtu.be/abc",
            "video_title": "Demo Video",
            "time_range": "00:01 - 00:10",
            "score": 0.91,
            "timestamped_url": "https://youtu.be/abc?t=1",
            "excerpt": "some transcript text",
        }
    ]
    monkeypatch.setattr(search_mod, "search_videos", lambda query, top_k=10: fake)

    runner = CliRunner()
    result = runner.invoke(cli_mod.cli, ["search", "foo"])

    assert result.exit_code == 0
    assert "Demo Video" in result.output


# --- BUG 3: --top-k must be >= 1 (IntRange rejects 0 / negative) ------------

def test_search_top_k_zero_rejected():
    runner = CliRunner()
    result = runner.invoke(cli_mod.cli, ["search", "foo", "--top-k", "0"])
    assert result.exit_code != 0


def test_search_top_k_negative_rejected():
    runner = CliRunner()
    result = runner.invoke(cli_mod.cli, ["search", "foo", "--top-k", "-3"])
    assert result.exit_code != 0


def test_search_top_k_valid_accepted(monkeypatch):
    seen = {}

    def capture(query, top_k=10):
        seen["top_k"] = top_k
        return []

    monkeypatch.setattr(search_mod, "search_videos", capture)

    runner = CliRunner()
    result = runner.invoke(cli_mod.cli, ["search", "foo", "--top-k", "5"])

    assert result.exit_code == 0
    assert seen["top_k"] == 5


# --- BUG 1: interactive must not crash on backend errors --------------------

def test_interactive_survives_backend_error(monkeypatch):
    """A RuntimeError from the backend prints a friendly message and the loop
    continues; feeding 'quit' afterwards exits cleanly (exit_code == 0)."""
    inputs = iter(["unreal nanite", "quit"])

    def fake_input(prompt=""):
        try:
            return next(inputs)
        except StopIteration:
            raise EOFError

    monkeypatch.setattr(cli_mod.console, "input", fake_input)

    def boom(query, top_k=10):
        raise RuntimeError("Qdrant down")

    monkeypatch.setattr(search_mod, "search_videos", boom)

    runner = CliRunner()
    result = runner.invoke(cli_mod.cli, ["interactive"])

    assert result.exit_code == 0
    assert "Search Error" in result.output
    assert "Goodbye" in result.output
    assert "Traceback" not in result.output


def test_interactive_survives_connection_error_via_eof(monkeypatch):
    """Even without an explicit 'quit', EOF ends the loop after an error."""
    inputs = iter(["query one"])

    def fake_input(prompt=""):
        try:
            return next(inputs)
        except StopIteration:
            raise EOFError

    monkeypatch.setattr(cli_mod.console, "input", fake_input)

    def boom(query, top_k=10):
        raise ConnectionError("Ollama down")

    monkeypatch.setattr(search_mod, "search_videos", boom)

    runner = CliRunner()
    result = runner.invoke(cli_mod.cli, ["interactive"])

    assert result.exit_code == 0
    assert "Connection Error" in result.output
    assert "Goodbye" in result.output
    assert "Traceback" not in result.output
