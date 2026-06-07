"""Tests for src.config env-var validation.

config reads the environment at IMPORT time, so we test by reloading the
module under a monkeypatched environment.
"""
import importlib

import pytest


def reload_config(monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import src.config as c
    return importlib.reload(c)


@pytest.fixture(autouse=True)
def _restore_config():
    """Restore src.config to default (unpatched) env after each test.

    monkeypatch auto-undoes the env changes; we reload once more here so other
    test modules importing src.config see a clean, default-valued module.
    """
    yield
    import src.config as c
    importlib.reload(c)


def test_valid_defaults_reload_ok(monkeypatch):
    c = reload_config(monkeypatch)
    assert c.MAX_AGE_YEARS == 3
    assert c.MIN_DURATION_SECONDS == 900
    assert c.EMBEDDING_DIM == 1024
    assert c.QDRANT_PORT == 6333
    assert c.CHUNK_DURATION_SECONDS == 120
    assert c.CHUNK_OVERLAP_SECONDS == 15
    assert all(isinstance(v, int) for v in (
        c.MAX_AGE_YEARS,
        c.MIN_DURATION_SECONDS,
        c.EMBEDDING_DIM,
        c.QDRANT_PORT,
        c.CHUNK_DURATION_SECONDS,
        c.CHUNK_OVERLAP_SECONDS,
    ))


def test_max_age_years_non_integer_raises(monkeypatch):
    with pytest.raises(ValueError) as exc:
        reload_config(monkeypatch, MAX_AGE_YEARS="three")
    assert "MAX_AGE_YEARS" in str(exc.value)
    assert "three" in str(exc.value)


def test_max_age_years_zero_raises(monkeypatch):
    with pytest.raises(ValueError) as exc:
        reload_config(monkeypatch, MAX_AGE_YEARS="0")
    assert "MAX_AGE_YEARS" in str(exc.value)


def test_max_age_years_negative_raises(monkeypatch):
    with pytest.raises(ValueError) as exc:
        reload_config(monkeypatch, MAX_AGE_YEARS="-2")
    assert "MAX_AGE_YEARS" in str(exc.value)


def test_min_duration_negative_raises(monkeypatch):
    with pytest.raises(ValueError) as exc:
        reload_config(monkeypatch, MIN_DURATION_SECONDS="-5")
    assert "MIN_DURATION_SECONDS" in str(exc.value)


def test_min_duration_zero_ok(monkeypatch):
    c = reload_config(monkeypatch, MIN_DURATION_SECONDS="0")
    assert c.MIN_DURATION_SECONDS == 0


def test_chunk_overlap_exceeds_duration_raises(monkeypatch):
    with pytest.raises(ValueError) as exc:
        reload_config(
            monkeypatch,
            CHUNK_OVERLAP_SECONDS="200",
            CHUNK_DURATION_SECONDS="120",
        )
    msg = str(exc.value)
    assert "CHUNK_OVERLAP_SECONDS" in msg
    assert "CHUNK_DURATION_SECONDS" in msg


def test_chunk_overlap_equal_duration_raises(monkeypatch):
    with pytest.raises(ValueError) as exc:
        reload_config(
            monkeypatch,
            CHUNK_OVERLAP_SECONDS="120",
            CHUNK_DURATION_SECONDS="120",
        )
    assert "CHUNK_OVERLAP_SECONDS" in str(exc.value)


def test_qdrant_port_too_high_raises(monkeypatch):
    with pytest.raises(ValueError) as exc:
        reload_config(monkeypatch, QDRANT_PORT="70000")
    assert "QDRANT_PORT" in str(exc.value)


def test_qdrant_port_negative_raises(monkeypatch):
    with pytest.raises(ValueError) as exc:
        reload_config(monkeypatch, QDRANT_PORT="-1")
    assert "QDRANT_PORT" in str(exc.value)


def test_embedding_dim_zero_raises(monkeypatch):
    with pytest.raises(ValueError) as exc:
        reload_config(monkeypatch, EMBEDDING_DIM="0")
    assert "EMBEDDING_DIM" in str(exc.value)


def test_config_error_is_value_error():
    import src.config as c
    assert issubclass(c.ConfigError, ValueError)
