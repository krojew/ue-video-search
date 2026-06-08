"""Orchestrates the full ingest pipeline: fetch → download → transcribe → embed → store."""

from __future__ import annotations

import gc
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import torch
from faster_whisper import WhisperModel
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

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

console = Console()


def run_fetch(
    use_cached: bool = True,
    skip_uefn: bool = True,
    skip_automotive: bool = True,
    skip_archvis: bool = True,
    include_streams: bool = True,
    force: bool = False,
) -> list[dict[str, Any]]:
    """Fetch (or load cached) video list from the channel.

    ``force=True`` bypasses the shrink guard in save_fetch_result, for a
    deliberate large catalog shrink (e.g. tightening filters or lowering
    MAX_AGE_YEARS) where the smaller list is intended to replace the cache so
    ``purge`` can act on it.
    """
    if use_cached:
        videos = load_video_list()
        if videos:
            console.print(f"[dim]Loaded {len(videos)} cached videos from disk.[/dim]")
            return videos

    console.print("[bold]Fetching video list from YouTube channel...[/bold]")
    fresh = fetch_video_list(
        skip_uefn=skip_uefn,
        skip_automotive=skip_automotive,
        skip_archvis=skip_archvis,
        include_streams=include_streams,
    )

    # save_fetch_result refuses to clobber a populated cache with an empty or
    # heavily-truncated fetch (which would also let `purge` delete valid
    # indexed videos). It returns whichever list is now authoritative on disk.
    # `force` overrides the guard for an intentional shrink.
    videos = save_fetch_result(fresh, force=force)
    if videos is not fresh:
        console.print(
            f"[yellow]Warning: fresh fetch returned {len(fresh)} video(s) but "
            f"{len(videos)} are cached on disk. Keeping the existing cache "
            f"instead of overwriting it with a likely-incomplete result. "
            f"Re-run with --force if this shrink is intentional.[/yellow]"
        )
    else:
        console.print(f"[green]Found {len(videos)} videos matching criteria.[/green]")
    return videos


def run_fetch_incremental(
    skip_uefn: bool = True,
    skip_automotive: bool = True,
    skip_archvis: bool = True,
    include_streams: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fetch fresh video list and merge with cached. Returns (full_list, new_videos)."""
    cached = load_video_list()

    console.print("[bold]Fetching latest video list from YouTube channel...[/bold]")
    fresh = fetch_video_list(
        skip_uefn=skip_uefn,
        skip_automotive=skip_automotive,
        skip_archvis=skip_archvis,
        include_streams=include_streams,
    )

    merged, new_only = merge_video_lists(cached, fresh)
    save_video_list(merged)

    if new_only:
        console.print(
            f"[green]Found {len(new_only)} new video(s)[/green] "
            f"[dim]({len(merged)} total, {len(cached)} previously cached)[/dim]"
        )
    else:
        console.print(f"[dim]No new videos found. {len(merged)} total on file.[/dim]")

    return merged, new_only


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


def _ingest_videos(
    videos: list[dict[str, Any]],
    skip_indexed: bool = True,
    label: str = "Processing videos",
) -> None:
    """Shared ingest logic: transcribe, embed, store a list of videos."""
    model: WhisperModel | None = None
    if not videos:
        console.print("[yellow]No videos to process.[/yellow]")
        return

    client = get_client()
    ensure_collection(client)

    # Pre-filter already-indexed videos to avoid loading Whisper unnecessarily.
    # Fetch the indexed-id set in one scroll instead of one round-trip per video.
    if skip_indexed:
        indexed_ids = list_indexed_video_ids(client)
        # Use .get() so a malformed entry missing video_id is dropped here
        # rather than raising KeyError and aborting the whole run before the
        # per-video loop (BUG5 — this pre-filter runs outside any try block).
        to_process = [v for v in videos if v.get("video_id") and v["video_id"] not in indexed_ids]
        skipped = len(videos) - len(to_process)
        if skipped:
            console.print(f"[dim]Skipping {skipped} already-indexed video(s).[/dim]")
        if not to_process:
            console.print("[dim]All videos already indexed. Nothing to do.[/dim]")
            return
    else:
        to_process = videos

    device = "cuda" if torch.cuda.is_available() else "cpu"

    console.print(f"\n[bold]Loading Whisper model ({WHISPER_MODEL}) on {device}...[/bold]")
    try:
        model = load_whisper_model()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress, ThreadPoolExecutor(max_workers=1) as downloader:
            task = progress.add_task(label, total=len(to_process))

            # Prefetch the first video's audio so transcription can start as
            # soon as the model is loaded. Subsequent prefetches happen inside
            # the loop, overlapping with the previous video's transcription.
            pending: Future[Any] | None = _submit_prefetch(downloader, to_process[0])

            for i, video in enumerate(to_process):
                # Snapshot the in-flight future for *this* video before swapping
                # `pending` to the next one — otherwise we would wait on the
                # wrong download. _submit_prefetch tolerates malformed entries
                # (returns None), so a bad NEXT video cannot abort this loop.
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
                    # counted as a failure and skipped instead of aborting the
                    # whole run.
                    vid = video["video_id"]
                    title = video["title"]
                    url = video["url"]

                    progress.update(task, description=f"[cyan]{title[:60]}[/cyan]")

                    # 1. Download audio + transcribe (audio is already on disk
                    #    if the prefetch landed; download_audio short-circuits)
                    segments = process_video(vid, url, model=model)
                    if not segments:
                        console.print(f"  [yellow]No segments for {vid}, skipping.[/yellow]")
                        progress.update(task, advance=1)
                        continue

                    texts = [build_chunk_embed_text(title, seg["text"]) for seg in segments]
                    embeddings = embed_texts(texts)

                    # 3. Store in Qdrant
                    count = upsert_chunks(vid, title, url, segments, embeddings, client)
                    console.print(f"  [green]✓ {title[:60]} — {count} chunks indexed[/green]")

                except Exception as e:
                    console.print(f"  [red]✗ malformed/failed video — {e}[/red]")

                progress.update(task, advance=1)
    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    console.print("\n[bold green]Ingest complete.[/bold green]")


def run_ingest(
    videos: list[dict[str, Any]] | None = None,
    skip_indexed: bool = True,
    skip_uefn: bool = True,
    skip_automotive: bool = True,
    skip_archvis: bool = True,
    include_streams: bool = True,
) -> None:
    """Full ingest pipeline for all videos."""
    if videos is None:
        videos = run_fetch(
            skip_uefn=skip_uefn,
            skip_automotive=skip_automotive,
            skip_archvis=skip_archvis,
            include_streams=include_streams,
        )
    _ingest_videos(videos, skip_indexed=skip_indexed, label="Processing videos")


def run_ingest_new_only(
    skip_uefn: bool = True,
    skip_automotive: bool = True,
    skip_archvis: bool = True,
    include_streams: bool = True,
) -> None:
    """Incremental ingest: process videos that are not yet in the index.

    The set of videos to process is (new videos) ∪ (cached videos missing
    from the Qdrant index). Cache membership is NOT used as the sole source
    of truth for "already done" — a video that was merged into the cache but
    failed to ingest stays eligible for retry because it is still absent from
    Qdrant. ``_ingest_videos(skip_indexed=True)`` performs the index diff, so
    passing the full merged list yields exactly that union.
    """
    merged, _new_only = run_fetch_incremental(
        skip_uefn=skip_uefn,
        skip_automotive=skip_automotive,
        skip_archvis=skip_archvis,
        include_streams=include_streams,
    )

    # Pass the full merged list (not just new_only). The skip-indexed filter
    # inside _ingest_videos keeps only videos absent from Qdrant, which is the
    # union of brand-new videos and previously-cached-but-unindexed ones.
    _ingest_videos(merged, skip_indexed=True, label="Processing new videos")
