"""Background ingest worker with event-based progress reporting.

Runs the ingest pipeline in a background thread and pushes status updates
to an asyncio.Queue so the web layer can stream them via SSE.
"""

from __future__ import annotations

import asyncio
import gc
import threading
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch
from faster_whisper import WhisperModel

from .config import WHISPER_MODEL
from .embeddings import build_chunk_embed_text, embed_texts
from .fetcher import fetch_video_list, load_video_list, merge_video_lists, save_video_list
from .transcriber import load_whisper_model, process_video
from .vectordb import ensure_collection, get_client, list_indexed_video_ids, upsert_chunks


class IngestPhase(str, Enum):
    IDLE = "idle"
    FETCHING = "fetching"
    LOADING_MODEL = "loading_model"
    PROCESSING = "processing"
    DONE = "done"
    ERROR = "error"


@dataclass
class IngestStatus:
    phase: IngestPhase = IngestPhase.IDLE
    total: int = 0
    completed: int = 0
    skipped: int = 0
    failed: int = 0
    current_video: str = ""
    message: str = ""
    new_videos_found: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase.value,
            "total": self.total,
            "completed": self.completed,
            "skipped": self.skipped,
            "failed": self.failed,
            "current_video": self.current_video,
            "message": self.message,
            "new_videos_found": self.new_videos_found,
        }


# Module-level state
_status = IngestStatus()
_lock = threading.Lock()
_event_queues: list[asyncio.Queue] = []
_event_loop: asyncio.AbstractEventLoop | None = None
_running = False
_QUEUE_MAXSIZE = 8


def get_status() -> dict[str, Any]:
    with _lock:
        return _status.to_dict()


def is_running() -> bool:
    return _running


def subscribe() -> asyncio.Queue:
    """Create a new SSE subscriber queue."""
    q: asyncio.Queue = asyncio.Queue(maxsize=_QUEUE_MAXSIZE)
    with _lock:
        _event_queues.append(q)
    return q


def unsubscribe(q: asyncio.Queue) -> None:
    with _lock:
        if q in _event_queues:
            _event_queues.remove(q)


def _emit(status: IngestStatus) -> None:
    """Push current status to all subscriber queues."""
    with _lock:
        data = status.to_dict()
        for q in _event_queues:
            try:
                if _event_loop and not _event_loop.is_closed():
                    _event_loop.call_soon_threadsafe(_push_latest, q, data)
            except Exception:
                pass


def _push_latest(q: asyncio.Queue, data: dict[str, Any]) -> None:
    """Keep only the most recent status events for slow SSE subscribers."""
    while True:
        try:
            q.put_nowait(data)
            return
        except asyncio.QueueFull:
            try:
                q.get_nowait()
            except asyncio.QueueEmpty:
                return


def _run_ingest(
    incremental: bool,
    reindex: bool,
    skip_uefn: bool = True,
    skip_automotive: bool = True,
    skip_archvis: bool = True,
    include_streams: bool = True,
) -> None:
    """Blocking ingest function meant to run in a thread."""
    global _running, _status
    model: WhisperModel | None = None
    _running = True
    _status = IngestStatus(phase=IngestPhase.FETCHING, message="Fetching video list from YouTube...")
    _emit(_status)

    try:
        # ── Fetch ──
        if incremental:
            cached = load_video_list()
            fresh = fetch_video_list(
                skip_uefn=skip_uefn,
                skip_automotive=skip_automotive,
                skip_archvis=skip_archvis,
                include_streams=include_streams,
            )
            merged, new_only = merge_video_lists(cached, fresh)
            save_video_list(merged)
            videos = new_only
            _status.new_videos_found = len(new_only)
            _status.message = f"Found {len(new_only)} new video(s) ({len(merged)} total)"
        else:
            videos = fetch_video_list(
                skip_uefn=skip_uefn,
                skip_automotive=skip_automotive,
                skip_archvis=skip_archvis,
                include_streams=include_streams,
            )
            save_video_list(videos)
            _status.new_videos_found = len(videos)
            _status.message = f"Found {len(videos)} videos matching criteria"

        _emit(_status)

        if not videos:
            _status.phase = IngestPhase.DONE
            _status.message = "No new videos to process."
            _emit(_status)
            return

        # ── Filter already indexed ──
        client = get_client()
        ensure_collection(client)

        skip_indexed = not reindex
        if skip_indexed:
            indexed_ids = list_indexed_video_ids(client)
            to_process = [v for v in videos if v["video_id"] not in indexed_ids]
            _status.skipped = len(videos) - len(to_process)
        else:
            to_process = videos

        _status.total = len(to_process)

        if not to_process:
            _status.phase = IngestPhase.DONE
            _status.message = f"All {len(videos)} videos already indexed. Nothing to do."
            _emit(_status)
            return

        # ── Load Whisper ──
        _status.phase = IngestPhase.LOADING_MODEL
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _status.message = f"Loading Whisper model ({WHISPER_MODEL}) on {device}..."
        _emit(_status)
        model = load_whisper_model()

        # ── Process ──
        _status.phase = IngestPhase.PROCESSING
        _emit(_status)

        for video in to_process:
            vid = video["video_id"]
            title = video["title"]
            url = video["url"]

            _status.current_video = title
            _status.message = f"Processing: {title[:80]}"
            _emit(_status)

            try:
                segments = process_video(vid, url, model=model)
                if not segments:
                    _status.failed += 1
                    _status.completed += 1
                    _emit(_status)
                    continue

                texts = [build_chunk_embed_text(title, seg["text"]) for seg in segments]
                embeddings = embed_texts(texts)
                count = upsert_chunks(vid, title, url, segments, embeddings, client)

                _status.completed += 1
                _status.message = f"Indexed: {title[:80]} ({count} chunks)"
                _emit(_status)

            except Exception as e:
                _status.failed += 1
                _status.completed += 1
                _status.message = f"Failed: {title[:60]} — {e}"
                _emit(_status)

        _status.phase = IngestPhase.DONE
        _status.current_video = ""
        _status.message = (
            f"Ingest complete. "
            f"{_status.completed - _status.failed} indexed, "
            f"{_status.failed} failed, "
            f"{_status.skipped} skipped."
        )
        _emit(_status)

    except Exception as e:
        _status.phase = IngestPhase.ERROR
        _status.message = f"Ingest error: {e}\n{traceback.format_exc()}"
        _emit(_status)
    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _running = False


def start_ingest(
    loop: asyncio.AbstractEventLoop,
    incremental: bool = False,
    reindex: bool = False,
    skip_uefn: bool = True,
    skip_automotive: bool = True,
    skip_archvis: bool = True,
    include_streams: bool = True,
) -> bool:
    """Start the ingest pipeline in a background thread. Returns False if already running."""
    global _event_loop
    if _running:
        return False
    _event_loop = loop
    t = threading.Thread(
        target=_run_ingest,
        args=(incremental, reindex, skip_uefn, skip_automotive, skip_archvis, include_streams),
        daemon=True,
    )
    t.start()
    return True
