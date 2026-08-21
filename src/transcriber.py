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

import av
import numpy as np
import torch
from faster_whisper import WhisperModel

from .config import (
    AUDIO_DIR,
    CHUNK_DURATION_SECONDS,
    CHUNK_OVERLAP_SECONDS,
    TRANSCRIBE_SLICE_OVERLAP_SECONDS,
    TRANSCRIBE_SLICE_SECONDS,
    TRANSCRIPT_DIR,
    WHISPER_MODEL,
)

# Sample rate Whisper expects; decoded slices are handed to the model directly
# as float32 arrays, so they must already be at this rate.
_SAMPLE_RATE = 16000


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


def _probe_duration(audio_path: Path) -> float | None:
    """Return the audio duration in seconds, or None if the container omits it.

    Uses PyAV's container metadata rather than decoding, so this is O(1) in file
    size. A None result means we cannot tell how many slices are needed, and the
    caller falls back to a single whole-file pass.
    """
    try:
        with av.open(str(audio_path), metadata_errors="ignore") as container:
            if container.duration is None:
                return None
            return container.duration / av.time_base
    except Exception:
        return None


def _decode_slice(audio_path: Path, start: float, duration: float) -> np.ndarray:
    """Decode ``[start, start+duration)`` as 16 kHz mono float32 via ffmpeg.

    Decoding a bounded span is the whole point: faster-whisper's own
    ``decode_audio`` only takes a file, which would put the entire track back in
    memory and defeat the slicing. ``-ss`` before ``-i`` makes ffmpeg seek on the
    input rather than decode-and-discard, so cost is proportional to the slice,
    not the offset. ffmpeg is already a hard dependency (yt-dlp's opus repack).
    """
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        raise RuntimeError(
            "ffmpeg not found on PATH; required to transcribe long audio in slices."
        )

    result = subprocess.run(
        [
            ffmpeg_bin,
            "-nostdin",
            "-loglevel", "error",
            "-ss", f"{start:.3f}",
            "-t", f"{duration:.3f}",
            "-i", str(audio_path),
            "-f", "s16le",
            "-ac", "1",
            "-ar", str(_SAMPLE_RATE),
            "-",
        ],
        check=True,
        capture_output=True,
        timeout=600,
    )
    # s16le → float32 in [-1, 1), matching what decode_audio would have produced.
    return np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def _run_model(
    audio: Any, model: WhisperModel, offset: float = 0.0
) -> list[dict[str, Any]]:
    """Transcribe one unit of audio (a path or a sample array) into raw segments.

    ``offset`` is added to every timestamp so a slice decoded from the middle of
    a file reports positions on the original file's timeline. Whisper's own
    timestamps are preserved otherwise — nothing is interpolated.
    """
    segments_iter, _info = model.transcribe(
        audio,
        language="en",
        initial_prompt=_WHISPER_PROMPT,
        word_timestamps=False,
        vad_filter=True,
    )
    return [
        {
            "start": float(s.start) + offset,
            "end": float(s.end) + offset,
            "text": s.text.strip(),
        }
        for s in segments_iter
        if s.text.strip()
    ]


def _transcribe_sliced(
    audio_path: Path, model: WhisperModel, duration: float
) -> list[dict[str, Any]]:
    """Transcribe long audio in bounded slices, concatenating the segments.

    Slice k nominally covers ``[k*S, (k+1)*S)`` but is decoded from
    ``k*S - overlap`` so the model has left context across the seam. Segments
    starting before the nominal boundary are dropped: the previous slice already
    emitted them, and this keeps the joined output free of duplicates while
    leaving no gap (the previous slice cannot emit past its own end).
    """
    raw_segments: list[dict[str, Any]] = []
    nominal_start = 0.0

    while nominal_start < duration:
        read_from = max(0.0, nominal_start - TRANSCRIBE_SLICE_OVERLAP_SECONDS)
        read_until = min(nominal_start + TRANSCRIBE_SLICE_SECONDS, duration)
        samples = _decode_slice(audio_path, read_from, read_until - read_from)

        if samples.size:
            for segment in _run_model(samples, model, offset=read_from):
                # First slice has no predecessor, so nothing to deduplicate.
                if nominal_start > 0 and segment["start"] < nominal_start:
                    continue
                raw_segments.append(segment)

        # Drop the slice's samples before decoding the next one, so two slices'
        # worth of audio are never resident at once.
        del samples
        nominal_start += TRANSCRIBE_SLICE_SECONDS

    return raw_segments


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

    Audio longer than TRANSCRIBE_SLICE_SECONDS is transcribed slice by slice
    rather than in one pass. faster-whisper holds the whole decoded track plus a
    log-mel spectrogram of it in memory, so a single pass costs RAM in
    proportion to video length (~8.7 GB for 2h20m) and silently OOM-kills a
    memory-capped container. Slicing keeps the peak flat; shorter audio takes
    the original single-pass path untouched.
    """
    if model is None:
        model = load_whisper_model()

    duration = _probe_duration(audio_path)
    if TRANSCRIBE_SLICE_SECONDS and duration is not None and duration > TRANSCRIBE_SLICE_SECONDS:
        raw_segments = _transcribe_sliced(audio_path, model, duration)
    else:
        # Hand the path straight to faster-whisper, which decodes it itself.
        raw_segments = _run_model(str(audio_path), model)

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
        # Do NOT persist an empty transcript. transcribe_audio returns [] only
        # for genuinely-silent videos (empty VAD output); genuine failures
        # (model load, decode errors) RAISE and propagate out through the
        # finally below, so they never reach save_transcript and are never
        # cached. Caching [] would make load_transcript hand back [] forever
        # (it is `not None`), so we skip the save for silence too — the only
        # cost is that truly-silent videos are retried every run.
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
