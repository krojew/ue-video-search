"""Download audio and transcribe with Whisper, preserving timestamps."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import whisper

from .config import AUDIO_DIR, TRANSCRIPT_DIR, WHISPER_MODEL


def download_audio(video_id: str, url: str) -> Path:
    """Download audio-only stream via yt-dlp. Returns path to the wav file."""
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    out_path = AUDIO_DIR / f"{video_id}.wav"
    if out_path.exists():
        return out_path

    # Prefer yt-dlp from PATH, otherwise fall back to the Python executable directory.
    yt_dlp_bin = shutil.which("yt-dlp")
    if yt_dlp_bin is None:
        yt_dlp_bin = str(Path(sys.executable).parent / "yt-dlp")

    # Download as wav 16kHz mono — optimal for Whisper
    subprocess.run(
        [
            yt_dlp_bin,
            "--no-playlist",
            "-x",
            "--audio-format", "wav",
            "--postprocessor-args", "ffmpeg:-ar 16000 -ac 1",
            "-o", out_path,
            url,
        ],
        check=True,
        capture_output=True,
    )
    return out_path


def transcribe_audio(audio_path: Path, model: whisper.Whisper | None = None) -> list[dict[str, Any]]:
    """Transcribe an audio file and return a list of segments with timestamps.

    Each segment dict has keys: start, end, text.
    """
    if model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper.load_model(WHISPER_MODEL, device=device)

    result = model.transcribe(
        str(audio_path),
        verbose=False,
        word_timestamps=False,
    )

    segments = []
    for seg in result.get("segments", []):
        text = seg["text"].strip()
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            continue

        seg_start = round(seg["start"], 2)
        seg_end = round(seg["end"], 2)
        duration = seg_end - seg_start
        total_chars = sum(len(s) for s in sentences)
        if total_chars == 0:
            continue

        current_time = seg_start
        for sentence in sentences:
            sentence_duration = duration * len(sentence) / total_chars
            segments.append(
                {
                    "start": round(current_time, 2),
                    "end": round(current_time + sentence_duration, 2),
                    "text": sentence,
                }
            )
            current_time += sentence_duration

    return segments


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


def process_video(video_id: str, url: str, model: whisper.Whisper | None = None) -> list[dict[str, Any]]:
    """Full pipeline: download audio → transcribe → save. Returns segments."""
    existing = load_transcript(video_id)
    if existing is not None:
        return existing

    audio_path = AUDIO_DIR / f"{video_id}.wav"
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
