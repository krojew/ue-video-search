"""Download audio and transcribe with Whisper, preserving timestamps."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch
from faster_whisper import WhisperModel

from .config import (
    AUDIO_DIR,
    CHUNK_DURATION_SECONDS,
    CHUNK_OVERLAP_SECONDS,
    TRANSCRIPT_DIR,
    WHISPER_MODEL,
)


def download_audio(video_id: str, url: str) -> Path:
    """Download audio-only stream via yt-dlp. Returns path to the audio file."""
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    out_path = AUDIO_DIR / f"{video_id}.opus"
    if out_path.exists():
        return out_path

    # Prefer yt-dlp from PATH, otherwise fall back to the Python executable directory.
    yt_dlp_bin = shutil.which("yt-dlp")
    if yt_dlp_bin is None:
        yt_dlp_bin = str(Path(sys.executable).parent / "yt-dlp")

    # YouTube serves audio as opus, so --audio-format opus is a no-op repack
    # rather than a transcode. Skipping the WAV resample to 16 kHz mono saves
    # bandwidth (~10x smaller files) and ffmpeg time; Whisper resamples
    # internally on load.
    try:
        subprocess.run(
            [
                yt_dlp_bin,
                "--no-playlist",
                "-x",
                "--audio-format", "opus",
                "-o", str(out_path),
                url,
            ],
            check=True,
            capture_output=True,
            timeout=600,
        )
    except subprocess.CalledProcessError as exc:
        # Surface yt-dlp's own diagnostic rather than an opaque non-zero exit.
        stderr = exc.stderr or b""
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        raise RuntimeError(
            f"yt-dlp failed for {video_id} (exit {exc.returncode}): {stderr.strip()}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"yt-dlp timed out after {exc.timeout}s downloading {video_id}"
        ) from exc
    return out_path


_WHISPER_PROMPT = (
    "Unreal Engine 5 tutorial. Topics include Nanite, Lumen, Niagara, Chaos, "
    "MetaHuman, MetaSounds, Megalights, World Partition, Geometry Script, PCG, "
    "Substrate, Blueprints, C++, materials, shaders, ray tracing, GPU, LOD, "
    "BSP, HLOD, post-process, virtual shadow maps, path tracing."
)


def _window_segments(
    segments: list[dict[str, Any]],
    target_duration: float,
    overlap: float,
) -> list[dict[str, Any]]:
    """Slide a fixed-duration window across timed segments, with overlap.

    Each output chunk preserves the real start/end of the contained segments;
    no timestamp interpolation. Adjacent windows share `overlap` seconds so
    queries that straddle a chunk boundary still match.
    """
    if not segments:
        return []

    stride = max(target_duration - overlap, 1.0)
    chunks: list[dict[str, Any]] = []
    i = 0
    n = len(segments)

    while i < n:
        window_start = segments[i]["start"]
        deadline = window_start + target_duration

        parts: list[str] = []
        j = i
        while j < n and segments[j]["start"] < deadline:
            parts.append(segments[j]["text"])
            j += 1

        if not parts:
            break

        last_end = segments[j - 1]["end"]
        text = " ".join(p for p in parts if p).strip()
        if text:
            chunks.append(
                {
                    "start": round(window_start, 2),
                    "end": round(last_end, 2),
                    "text": text,
                }
            )

        if j >= n:
            break

        target_next = window_start + stride
        new_i = i + 1
        while new_i < n and segments[new_i]["start"] < target_next:
            new_i += 1
        i = new_i if new_i > i else i + 1

    return chunks


def load_whisper_model() -> WhisperModel:
    """Load Whisper using the best device + compute_type for the current host.

    CUDA gets float16; CPU gets int8 (CTranslate2 quantizes at load time).
    """
    if torch.cuda.is_available():
        return WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16")
    return WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")


def transcribe_audio(audio_path: Path, model: WhisperModel | None = None) -> list[dict[str, Any]]:
    """Transcribe an audio file and return a list of windowed chunks.

    Each chunk dict has keys: start, end, text. Chunks are produced by sliding
    a CHUNK_DURATION_SECONDS window over Whisper's native segments with
    CHUNK_OVERLAP_SECONDS of overlap between adjacent windows. Timestamps are
    Whisper's own — not interpolated.
    """
    if model is None:
        model = load_whisper_model()

    segments_iter, _info = model.transcribe(
        str(audio_path),
        language="en",
        initial_prompt=_WHISPER_PROMPT,
        word_timestamps=False,
        vad_filter=True,
    )

    raw_segments = [
        {"start": float(s.start), "end": float(s.end), "text": s.text.strip()}
        for s in segments_iter
        if s.text.strip()
    ]
    if not raw_segments:
        return []

    return _window_segments(raw_segments, CHUNK_DURATION_SECONDS, CHUNK_OVERLAP_SECONDS)


def save_transcript(video_id: str, segments: list[dict[str, Any]]) -> Path:
    """Save transcript segments to a JSON file atomically and durably.

    Writes to a unique temp file in the same directory, flushes + fsyncs it,
    then os.replace()s it into place. os.replace is atomic on the same
    filesystem, so a reader always sees either the old complete file or the new
    complete one (no torn reads), and the fsync narrows the power-loss window in
    which the rename could land before the data. The temp file is cleaned up if
    anything fails before the replace.
    """
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    path = TRANSCRIPT_DIR / f"{video_id}.json"
    data = json.dumps(segments, indent=2)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(TRANSCRIPT_DIR), prefix=path.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return path


def load_transcript(video_id: str) -> list[dict[str, Any]] | None:
    """Load a previously saved transcript, or None if not found.

    A corrupt/unparseable file is treated as not-cached (returns None) so the
    video gets re-processed rather than raising on every load.
    """
    path = TRANSCRIPT_DIR / f"{video_id}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def process_video(
    video_id: str,
    url: str,
    model: WhisperModel | None = None,
    cleanup_audio: bool = True,
) -> list[dict[str, Any]]:
    """Full pipeline: download audio → transcribe → save. Returns segments.

    When ``cleanup_audio`` is True (the default), the downloaded ``.opus`` file
    is deleted after transcription regardless of whether it pre-existed on disk.
    This matters because a prefetch step downloads the *next* video's audio
    ahead of time, so by the time this runs the file is normally already
    present — keying cleanup on pre-existence (the old behavior) would leak
    every prefetched file and grow disk usage without bound. The
    transcript-cache short-circuit below never touches the filesystem, so a
    cached run leaves any audio untouched.
    """
    existing = load_transcript(video_id)
    if existing is not None:
        return existing

    audio_path = download_audio(video_id, url)
    try:
        segments = transcribe_audio(audio_path, model=model)
        # Do NOT persist an empty transcript. transcribe_audio returns [] both
        # for genuinely-silent videos and for upstream failures; caching []
        # would make load_transcript hand back [] forever (it is `not None`),
        # permanently blocking re-ingest of a recoverable failure. Skipping the
        # save means truly-silent videos are retried every run — an acceptable
        # cost versus losing recoverable ones.
        if segments:
            save_transcript(video_id, segments)
    finally:
        # Best-effort: remove the audio whether or not it pre-existed, since
        # prefetch makes "already on disk" the normal case (see docstring).
        if cleanup_audio:
            try:
                audio_path.unlink(missing_ok=True)
            except OSError:
                pass  # Ignore deletion errors

    return segments
