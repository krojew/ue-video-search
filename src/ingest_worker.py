"""Background ingest worker with event-based progress reporting.

Runs the ingest pipeline in a background thread and pushes status updates
to an asyncio.Queue so the web layer can stream them via SSE.
"""

from __future__ import annotations

import asyncio
import gc
import threading
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch
from faster_whisper import WhisperModel

from .config import WHISPER_MODEL
from .embeddings import build_chunk_embed_text, embed_texts
from .fetcher import (
    fetch_video_list,
    load_video_list,
    merge_video_lists,
    save_fetch_result,
    save_video_list,
)
from .transcriber import download_audio, load_transcript, load_whisper_model, process_video
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
    with _lock:
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


def _emit_locked(data: dict[str, Any]) -> None:
    """Dispatch a prebuilt status snapshot to all subscriber queues.

    The caller MUST already hold ``_lock`` (so the queue list and the snapshot
    are consistent and we do not re-enter the non-reentrant lock).
    """
    for q in _event_queues:
        try:
            if _event_loop and not _event_loop.is_closed():
                _event_loop.call_soon_threadsafe(_push_latest, q, data)
        except Exception:
            pass





def _update_status(*, inc: dict[str, int] | None = None, **fields: Any) -> None:
    """Atomically update ``_status`` fields and emit the resulting snapshot.

    Both the mutation and the snapshot happen under ``_lock`` so concurrent
    readers (``get_status``) can never observe a torn/inconsistent state
    (BUG4). ``fields`` sets absolute values; ``inc`` applies integer deltas
    (read-modify-write) to counter fields under the same lock.
    """
    with _lock:
        for key, value in fields.items():
            setattr(_status, key, value)
        if inc:
            for key, delta in inc.items():
                setattr(_status, key, getattr(_status, key) + delta)
        _emit_locked(_status.to_dict())


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


def _submit_prefetch(
    pool: ThreadPoolExecutor, video: dict[str, Any]
) -> Future[Any] | None:
    """Queue an audio download for `video` if it is not already cached.

    Returns None when the transcript already exists, since process_video
    will short-circuit and the audio would never be read. Also returns None
    for a malformed entry (missing video_id/url) so a bad dict cannot abort
    the prefetch of an otherwise healthy run — the malformed video is handled
    (and counted) in the main loop's per-video error handling.
    """
    try:
        video_id = video["video_id"]
        url = video["url"]
    except (KeyError, TypeError):
        return None
    if load_transcript(video_id) is not None:
        return None
    return pool.submit(download_audio, video_id, url)


def _run_ingest(
    incremental: bool,
    reindex: bool,
    skip_uefn: bool = True,
    skip_automotive: bool = True,
    skip_archvis: bool = True,
    include_streams: bool = True,
) -> None:
    """Blocking ingest function meant to run in a thread.

    `_running` is set to True by `start_ingest` under `_lock` *before* the
    thread is launched (see BUG3). This function only clears it in `finally`.
    """
    global _running, _status
    model: WhisperModel | None = None
    with _lock:
        _status = IngestStatus(
            phase=IngestPhase.FETCHING,
            message="Fetching video list from YouTube...",
        )
        _emit_locked(_status.to_dict())

    try:
        # ── Fetch ──
        # `incremental` and `reindex` are mutually exclusive intents: incremental
        # means "process what isn't indexed yet", reindex means "re-process
        # everything". The web endpoint can supply both; treat incremental as
        # the winner (ignore reindex) so the confusing combination can't
        # silently skip the videos a caller asked to reindex.
        if incremental:
            reindex = False
        skip_indexed = not reindex
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
            # Candidate pool: when skip-indexed is on, consider the *whole*
            # merged list so the index diff below also picks up previously
            # cached videos that failed to ingest (absent from Qdrant). Cache
            # membership alone must NOT mark a video as done — Qdrant is the
            # source of truth. With reindex (skip_indexed off) there is no
            # index diff, so fall back to processing only the brand-new ones.
            videos = merged if skip_indexed else new_only
            _update_status(
                new_videos_found=len(new_only),
                message=f"Found {len(new_only)} new video(s) ({len(merged)} total)",
            )
        else:
            fresh = fetch_video_list(
                skip_uefn=skip_uefn,
                skip_automotive=skip_automotive,
                skip_archvis=skip_archvis,
                include_streams=include_streams,
            )
            # Guard against a transient empty/truncated fetch clobbering the
            # cache (which would also let `purge` delete valid indexed videos).
            # Shared with pipeline.run_fetch so the two paths cannot drift.
            videos = save_fetch_result(fresh)
            _update_status(
                new_videos_found=len(videos),
                message=f"Found {len(videos)} videos matching criteria",
            )

        if not videos:
            _update_status(
                phase=IngestPhase.DONE,
                message="No new videos to process.",
            )
            return

        # ── Filter already indexed ──
        client = get_client()
        ensure_collection(client)

        if skip_indexed:
            indexed_ids = list_indexed_video_ids(client)
            # Use .get() so a malformed entry missing video_id is dropped here
            # rather than raising KeyError outside the per-video try and
            # flipping the whole run to ERROR (BUG5).
            to_process = [v for v in videos if v.get("video_id") and v["video_id"] not in indexed_ids]
            skipped = len(videos) - len(to_process)
        else:
            to_process = videos
            skipped = 0

        _update_status(skipped=skipped, total=len(to_process))

        if not to_process:
            _update_status(
                phase=IngestPhase.DONE,
                message=f"All {len(videos)} videos already indexed. Nothing to do.",
            )
            return

        # ── Load Whisper ──
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _update_status(
            phase=IngestPhase.LOADING_MODEL,
            message=f"Loading Whisper model ({WHISPER_MODEL}) on {device}...",
        )
        model = load_whisper_model()

        # ── Process ──
        _update_status(phase=IngestPhase.PROCESSING)

        with ThreadPoolExecutor(max_workers=1) as downloader:
            # Prefetch the first video's audio so the worker thread starts
            # downloading while Whisper is mid-load. Subsequent prefetches
            # happen inside the loop, overlapping with the previous video's
            # transcription.
            pending: Future[Any] | None = _submit_prefetch(downloader, to_process[0])

            # Videos with no detectable speech: counted as skipped (not failed),
            # tracked locally so the "indexed" summary math stays correct.
            silent = 0

            for i, video in enumerate(to_process):
                # Snapshot the in-flight future for *this* video before
                # swapping `pending` to the next one. _submit_prefetch
                # tolerates malformed entries (returns None), so a bad NEXT
                # video cannot abort this loop.
                current_pending = pending
                pending = (
                    _submit_prefetch(downloader, to_process[i + 1])
                    if i + 1 < len(to_process)
                    else None
                )

                if current_pending is not None:
                    try:
                        current_pending.result()
                    except Exception:
                        pass  # process_video will re-raise from its own download attempt

                try:
                    # Key access is inside the try so a malformed entry is
                    # counted as failed and skipped instead of aborting the run.
                    vid = video["video_id"]
                    title = video["title"]
                    url = video["url"]

                    _update_status(
                        current_video=title,
                        message=f"Processing: {title[:80]}",
                    )

                    segments = process_video(vid, url, model=model)
                    if not segments:
                        # No speech detected (not a failure — process_video
                        # returns [] for silent videos by design and they are
                        # retry-eligible next run). Count as skipped, matching
                        # the CLI, and advance progress without bumping failed.
                        silent += 1
                        _update_status(
                            inc={"completed": 1, "skipped": 1},
                            message=f"No speech detected: {title[:80]} (will retry next run)",
                        )
                        continue

                    texts = [build_chunk_embed_text(title, seg["text"]) for seg in segments]
                    embeddings = embed_texts(texts)
                    count = upsert_chunks(vid, title, url, segments, embeddings, client)

                    _update_status(
                        inc={"completed": 1},
                        message=f"Indexed: {title[:80]} ({count} chunks)",
                    )

                except Exception as e:
                    _update_status(
                        inc={"failed": 1, "completed": 1},
                        message=f"Failed video — {e}",
                    )

        # Single writer at this point (the loop is done), so reading a locked
        # snapshot for the summary cannot race with another mutation.
        snap = get_status()
        _update_status(
            phase=IngestPhase.DONE,
            current_video="",
            message=(
                f"Ingest complete. "
                f"{snap['completed'] - snap['failed'] - silent} indexed, "
                f"{snap['failed']} failed, "
                f"{snap['skipped']} skipped."
            ),
        )

    except Exception as e:
        _update_status(
            phase=IngestPhase.ERROR,
            message=f"Ingest error: {e}\n{traceback.format_exc()}",
        )
    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        with _lock:
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
    global _event_loop, _running
    # Atomic check-and-set under _lock: two near-simultaneous callers must not
    # both observe _running == False and start two ingest threads (BUG3).
    with _lock:
        if _running:
            return False
        _running = True
    _event_loop = loop
    # Construct AND start the thread inside the try: if Thread.__init__ raises
    # (e.g. resource exhaustion) the reservation must still be rolled back,
    # otherwise _running stays True and every future ingest is wrongly refused.
    try:
        t = threading.Thread(
            target=_run_ingest,
            args=(incremental, reindex, skip_uefn, skip_automotive, skip_archvis, include_streams),
            daemon=True,
        )
        t.start()
    except Exception:
        # Roll back the reservation if the thread never actually launched.
        with _lock:
            _running = False
        raise
    return True
