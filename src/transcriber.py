"""Download audio and transcribe with Whisper, preserving timestamps."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
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
    subprocess.run(
        [
            yt_dlp_bin,
            "--no-playlist",
            "-x",
            "--audio-format", "opus",
            "-o", out_path,
            url,
        ],
        check=True,
        capture_output=True,
    )
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
    """Save transcript segments to a JSON file."""
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    path = TRANSCRIPT_DIR / f"{video_id}.json"
    path.write_text(json.dumps(segments, indent=2))
    return path


def load_transcript(video_id: str) -> list[dict[str, Any]] | None:
    """Load a previously saved transcript, or None if not found."""
    path = TRANSCRIPT_DIR / f"{video_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def process_video(video_id: str, url: str, model: WhisperModel | None = None) -> list[dict[str, Any]]:
    """Full pipeline: download audio → transcribe → save. Returns segments."""
    existing = load_transcript(video_id)
    if existing is not None:
        return existing

    audio_path = AUDIO_DIR / f"{video_id}.opus"
    audio_existed = audio_path.exists()
    audio_path = download_audio(video_id, url)
    segments = transcribe_audio(audio_path, model=model)
    save_transcript(video_id, segments)

    # Delete temporary audio file if it was downloaded (not cached)
    if not audio_existed:
        try:
            audio_path.unlink(missing_ok=True)
        except Exception:
            pass  # Ignore deletion errors

    return segments
