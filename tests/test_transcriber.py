"""Hermetic tests for src.transcriber.

No real yt-dlp, Whisper, network or GPU are touched — every heavy dependency
is monkeypatched, and AUDIO_DIR / TRANSCRIPT_DIR are redirected at tmp dirs.
"""

import pytest

import src.transcriber as transcriber


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    """Point AUDIO_DIR and TRANSCRIPT_DIR at isolated tmp dirs."""
    audio = tmp_path / "audio"
    transcripts = tmp_path / "transcripts"
    audio.mkdir()
    transcripts.mkdir()
    monkeypatch.setattr(transcriber, "AUDIO_DIR", audio)
    monkeypatch.setattr(transcriber, "TRANSCRIPT_DIR", transcripts)
    return audio, transcripts


# ── BUG 1: prefetched audio must still be cleaned up ──────────────────────


def test_process_video_deletes_preexisting_audio(dirs, monkeypatch):
    """A .opus already on disk (the prefetch case) MUST be deleted afterward."""
    audio_dir, _ = dirs
    opus = audio_dir / "vid1.opus"
    opus.write_bytes(b"fake audio")
    assert opus.exists()

    monkeypatch.setattr(transcriber, "download_audio", lambda vid, url: opus)
    monkeypatch.setattr(
        transcriber,
        "transcribe_audio",
        lambda path, model=None: [{"start": 0.0, "end": 1.0, "text": "hi"}],
    )

    result = transcriber.process_video("vid1", "http://x", cleanup_audio=True)

    assert result == [{"start": 0.0, "end": 1.0, "text": "hi"}]
    assert not opus.exists(), "pre-existing prefetched audio should be deleted"


def test_process_video_transcript_cache_leaves_files_untouched(dirs, monkeypatch):
    """The transcript-exists early-return must not touch any files."""
    audio_dir, _ = dirs
    opus = audio_dir / "vid2.opus"
    opus.write_bytes(b"fake audio")

    transcriber.save_transcript("vid2", [{"start": 0.0, "end": 1.0, "text": "cached"}])

    def _boom(*a, **k):  # pragma: no cover - must never run
        raise AssertionError("download/transcribe should not run on cache hit")

    monkeypatch.setattr(transcriber, "download_audio", _boom)
    monkeypatch.setattr(transcriber, "transcribe_audio", _boom)

    result = transcriber.process_video("vid2", "http://x")

    assert result == [{"start": 0.0, "end": 1.0, "text": "cached"}]
    assert opus.exists(), "cache hit must not delete unrelated audio"


def test_process_video_no_cleanup_when_disabled(dirs, monkeypatch):
    """cleanup_audio=False keeps the audio on disk."""
    audio_dir, _ = dirs
    opus = audio_dir / "vid3.opus"
    opus.write_bytes(b"fake audio")

    monkeypatch.setattr(transcriber, "download_audio", lambda vid, url: opus)
    monkeypatch.setattr(
        transcriber,
        "transcribe_audio",
        lambda path, model=None: [{"start": 0.0, "end": 1.0, "text": "hi"}],
    )

    transcriber.process_video("vid3", "http://x", cleanup_audio=False)

    assert opus.exists()


# ── BUG 2: empty transcripts must not be cached ───────────────────────────


def test_empty_transcript_not_cached_and_retried(dirs, monkeypatch):
    """[] from transcribe_audio must not be persisted, so later runs retry."""
    audio_dir, transcript_dir = dirs
    opus = audio_dir / "vid_empty.opus"
    opus.write_bytes(b"fake audio")

    monkeypatch.setattr(transcriber, "download_audio", lambda vid, url: opus)
    monkeypatch.setattr(transcriber, "transcribe_audio", lambda path, model=None: [])

    result = transcriber.process_video("vid_empty", "http://x")

    assert result == []
    # No transcript file was written.
    assert not (transcript_dir / "vid_empty.json").exists()
    # A subsequent run sees no cache and would process again.
    assert transcriber.load_transcript("vid_empty") is None


# ── BUG 3: atomic save + corruption-tolerant load ─────────────────────────


def test_save_transcript_round_trips(dirs):
    segments = [{"start": 0.0, "end": 2.5, "text": "hello world"}]
    transcriber.save_transcript("vidrt", segments)
    assert transcriber.load_transcript("vidrt") == segments


def test_save_transcript_leaves_no_temp_file(dirs):
    _, transcript_dir = dirs
    transcriber.save_transcript("vidtmp", [{"start": 0.0, "end": 1.0, "text": "x"}])
    assert (transcript_dir / "vidtmp.json").exists()
    assert not (transcript_dir / "vidtmp.json.tmp").exists()


def test_load_transcript_corrupt_returns_none(dirs):
    _, transcript_dir = dirs
    (transcript_dir / "vidbad.json").write_text("{ this is not valid json ")
    assert transcriber.load_transcript("vidbad") is None
