"""Hermetic tests for src.transcriber.

No real yt-dlp, Whisper, network or GPU are touched — every heavy dependency
is monkeypatched, and AUDIO_DIR / TRANSCRIPT_DIR are redirected at tmp dirs.
"""

import subprocess
from pathlib import Path

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


def test_process_video_cleans_audio_when_transcription_raises(dirs, monkeypatch):
    """The finally block must delete the audio even when transcribe_audio RAISES.

    Regression guard for the headline contract: moving the unlink out of the
    `finally` (into the try, or after save) would leak audio on any failure.
    """
    audio_dir, transcript_dir = dirs
    opus = audio_dir / "vid_boom.opus"
    opus.write_bytes(b"fake audio")

    monkeypatch.setattr(transcriber, "download_audio", lambda vid, url: opus)

    def boom(path, model=None):
        raise RuntimeError("decode failed")

    monkeypatch.setattr(transcriber, "transcribe_audio", boom)

    with pytest.raises(RuntimeError):
        transcriber.process_video("vid_boom", "http://x")

    # finally ran: audio deleted, and the failure was NOT cached as a transcript.
    assert not opus.exists(), "audio must be cleaned up even when transcription raises"
    assert not (transcript_dir / "vid_boom.json").exists()


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
    # Audio cleanup must still run on the empty-transcript path (cleanup is
    # independent of whether the transcript was cached).
    assert not opus.exists(), "audio must be cleaned up even for an empty transcript"


# ── BUG 3: atomic save + corruption-tolerant load ─────────────────────────


def test_save_transcript_round_trips(dirs):
    segments = [{"start": 0.0, "end": 2.5, "text": "hello world"}]
    transcriber.save_transcript("vidrt", segments)
    assert transcriber.load_transcript("vidrt") == segments


def test_save_transcript_leaves_no_temp_file(dirs):
    _, transcript_dir = dirs
    transcriber.save_transcript("vidtmp", [{"start": 0.0, "end": 1.0, "text": "x"}])
    # mkstemp inserts a random infix, so assert on the full directory contents
    # rather than a fixed '.tmp' name that mkstemp would never produce.
    assert [p.name for p in transcript_dir.iterdir()] == ["vidtmp.json"]


def test_save_transcript_cleans_temp_on_failure(dirs, monkeypatch):
    """If the atomic write fails mid-way, no *.tmp leftover should remain."""
    _, transcript_dir = dirs

    def boom(*a, **k):
        raise RuntimeError("fsync failed")

    monkeypatch.setattr(transcriber.os, "fsync", boom)
    with pytest.raises(RuntimeError):
        transcriber.save_transcript("vidfail", [{"start": 0.0, "end": 1.0, "text": "x"}])

    leftovers = [p.name for p in transcript_dir.iterdir()]
    assert leftovers == [], f"temp file not cleaned up: {leftovers}"


def test_load_transcript_corrupt_returns_none(dirs):
    _, transcript_dir = dirs
    (transcript_dir / "vidbad.json").write_text("{ this is not valid json ")
    assert transcriber.load_transcript("vidbad") is None


# ── BUG 4: yt-dlp timeout + stderr surfacing ──────────────────────────────


def test_download_audio_called_process_error_surfaces_stderr(dirs, monkeypatch):
    """CalledProcessError must become a RuntimeError carrying yt-dlp's stderr."""
    monkeypatch.setattr(transcriber.shutil, "which", lambda name: "/usr/bin/yt-dlp")

    def _raise(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0], stderr=b"boom")

    monkeypatch.setattr(transcriber.subprocess, "run", _raise)

    with pytest.raises(RuntimeError, match="boom"):
        transcriber.download_audio("viderr", "http://x")


def test_download_audio_timeout_raises_runtime_error(dirs, monkeypatch):
    """TimeoutExpired must become a clear RuntimeError."""
    monkeypatch.setattr(transcriber.shutil, "which", lambda name: "/usr/bin/yt-dlp")

    def _raise(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], timeout=600)

    monkeypatch.setattr(transcriber.subprocess, "run", _raise)

    with pytest.raises(RuntimeError, match="timed out"):
        transcriber.download_audio("vidto", "http://x")


def test_download_audio_passes_timeout(dirs, monkeypatch):
    """The subprocess.run call must include a timeout."""
    monkeypatch.setattr(transcriber.shutil, "which", lambda name: "/usr/bin/yt-dlp")
    captured = {}

    def _run(*args, **kwargs):
        captured.update(kwargs)

        class _R:
            pass

        return _R()

    monkeypatch.setattr(transcriber.subprocess, "run", _run)

    transcriber.download_audio("vidok", "http://x")
    assert captured.get("timeout") is not None


# ── Long audio must be transcribed in bounded slices (OOM guard) ──────────
#
# faster-whisper holds the whole decoded track plus a log-mel spectrogram of it
# in memory, so a single pass over a 2h+ video peaked at ~8.7 GB RSS and got the
# 8 GB container OOM-killed mid-run. The kill raises no Python exception, so the
# ingest worker never reported it and every video queued behind it was dropped.


class _FakeSegment:
    def __init__(self, start, end, text):
        self.start = start
        self.end = end
        self.text = text


class _FakeModel:
    """Records what it was handed and replays canned segments per call."""

    def __init__(self, per_call):
        self.per_call = list(per_call)
        self.calls = []

    def transcribe(self, audio, **kwargs):
        self.calls.append(audio)
        segments = self.per_call.pop(0) if self.per_call else []
        return iter([_FakeSegment(*s) for s in segments]), None


def _slice_env(monkeypatch, *, slice_secs, overlap, duration, decoded_len=16000):
    """Configure slicing and stub out duration probing / ffmpeg decoding."""
    monkeypatch.setattr(transcriber, "TRANSCRIBE_SLICE_SECONDS", slice_secs)
    monkeypatch.setattr(transcriber, "TRANSCRIBE_SLICE_OVERLAP_SECONDS", overlap)
    monkeypatch.setattr(transcriber, "_probe_duration", lambda p: duration)

    decoded = []

    def _fake_decode(path, start, dur):
        decoded.append((round(start, 3), round(dur, 3)))
        return transcriber.np.zeros(decoded_len, dtype=transcriber.np.float32)

    monkeypatch.setattr(transcriber, "_decode_slice", _fake_decode)
    return decoded


def test_short_audio_still_takes_the_single_pass_path(monkeypatch):
    """Audio under the slice length must hand the PATH to faster-whisper, unchanged."""
    _slice_env(monkeypatch, slice_secs=1200, overlap=5, duration=600.0)
    model = _FakeModel([[(0.0, 2.0, "hello")]])

    out = transcriber.transcribe_audio(Path("/tmp/short.opus"), model=model)

    assert model.calls == ["/tmp/short.opus"], "short audio should not be sliced"
    assert out and out[0]["text"] == "hello"


def test_long_audio_is_decoded_in_bounded_slices(monkeypatch):
    """Each ffmpeg decode must cover at most one slice, never the whole file."""
    decoded = _slice_env(monkeypatch, slice_secs=1200, overlap=5, duration=3000.0)
    model = _FakeModel([[(0.0, 1.0, "a")], [(0.0, 1.0, "b")], [(0.0, 1.0, "c")]])

    transcriber.transcribe_audio(Path("/tmp/long.opus"), model=model)

    # 3000s at 1200s slices → 3 slices; each reads its own span plus left overlap.
    assert decoded == [(0.0, 1200.0), (1195.0, 1205.0), (2395.0, 605.0)]
    assert all(dur <= 1200.0 + 5 for _start, dur in decoded), "slice exceeded its bound"
    # The model must receive sample arrays, never the whole-file path.
    assert all(not isinstance(c, str) for c in model.calls)


def test_slice_timestamps_are_offset_onto_the_file_timeline(monkeypatch):
    """A segment found inside slice 2 must report its position in the whole file."""
    _slice_env(monkeypatch, slice_secs=1200, overlap=5, duration=2000.0)
    # Slice 2 reads from 1195.0; a hit 10s in sits at 1205.0 on the file timeline.
    model = _FakeModel([[(0.0, 1.0, "first")], [(10.0, 20.0, "second")]])

    out = transcriber.transcribe_audio(Path("/tmp/long.opus"), model=model)

    texts = " ".join(c["text"] for c in out)
    assert "second" in texts
    assert any(abs(c["start"] - 1205.0) < 0.01 or abs(c["end"] - 1215.0) < 0.01 for c in out), (
        f"slice-2 timestamps were not offset onto the file timeline: {out}"
    )


def test_overlap_region_is_not_emitted_twice(monkeypatch):
    """Re-read overlap gives left context only; it must not duplicate segments."""
    _slice_env(monkeypatch, slice_secs=1200, overlap=5, duration=2000.0)
    # Slice 2 starts reading at 1195.0 and re-hears the tail slice 1 already
    # emitted (offset 0.0-3.0 → 1195.0-1198.0, before the 1200.0 boundary).
    model = _FakeModel(
        [
            [(1190.0, 1198.0, "tail")],
            [(0.0, 3.0, "tail"), (10.0, 12.0, "fresh")],
        ]
    )

    out = transcriber.transcribe_audio(Path("/tmp/long.opus"), model=model)

    joined = " ".join(c["text"] for c in out)
    assert joined.count("tail") == 1, f"overlap duplicated a segment: {joined}"
    assert "fresh" in joined


def test_slicing_disabled_falls_back_to_single_pass(monkeypatch):
    """TRANSCRIBE_SLICE_SECONDS=0 must restore the original whole-file behaviour."""
    _slice_env(monkeypatch, slice_secs=0, overlap=0, duration=99999.0)
    model = _FakeModel([[(0.0, 1.0, "whole")]])

    transcriber.transcribe_audio(Path("/tmp/long.opus"), model=model)

    assert model.calls == ["/tmp/long.opus"]


def test_unknown_duration_falls_back_to_single_pass(monkeypatch):
    """If the container has no duration we cannot slice; do not guess."""
    _slice_env(monkeypatch, slice_secs=1200, overlap=5, duration=None)
    model = _FakeModel([[(0.0, 1.0, "whole")]])

    transcriber.transcribe_audio(Path("/tmp/long.opus"), model=model)

    assert model.calls == ["/tmp/long.opus"]


def test_decode_slice_requests_a_bounded_span_from_ffmpeg(monkeypatch):
    """_decode_slice must pass -ss/-t so ffmpeg never decodes the whole file."""
    monkeypatch.setattr(transcriber.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    captured = {}

    def _run(cmd, **kwargs):
        captured["cmd"] = cmd

        class _R:
            stdout = b"\x00\x00" * 100

        return _R()

    monkeypatch.setattr(transcriber.subprocess, "run", _run)

    samples = transcriber._decode_slice(Path("/tmp/a.opus"), 90.0, 30.0)

    cmd = captured["cmd"]
    assert "-ss" in cmd and cmd[cmd.index("-ss") + 1] == "90.000"
    assert "-t" in cmd and cmd[cmd.index("-t") + 1] == "30.000"
    # -ss must precede -i, otherwise ffmpeg decodes and discards everything first.
    assert cmd.index("-ss") < cmd.index("-i")
    assert samples.dtype == transcriber.np.float32
    assert len(samples) == 100
