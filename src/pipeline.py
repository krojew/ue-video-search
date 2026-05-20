"""Orchestrates the full ingest pipeline: fetch → download → transcribe → embed → store."""

from __future__ import annotations

import gc
from typing import Any

import torch
import whisper
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from .config import WHISPER_MODEL
from .embeddings import build_chunk_embed_text, embed_texts
from .fetcher import fetch_video_list, load_video_list, merge_video_lists, save_video_list
from .transcriber import process_video
from .vectordb import ensure_collection, get_client, upsert_chunks, video_already_indexed

console = Console()


def run_fetch(
    use_cached: bool = True,
    skip_uefn: bool = True,
    skip_automotive: bool = True,
    skip_archvis: bool = True,
    include_streams: bool = True,
) -> list[dict[str, Any]]:
    """Fetch (or load cached) video list from the channel."""
    if use_cached:
        videos = load_video_list()
        if videos:
            console.print(f"[dim]Loaded {len(videos)} cached videos from disk.[/dim]")
            return videos

    console.print("[bold]Fetching video list from YouTube channel...[/bold]")
    videos = fetch_video_list(
        skip_uefn=skip_uefn,
        skip_automotive=skip_automotive,
        skip_archvis=skip_archvis,
        include_streams=include_streams,
    )
    save_video_list(videos)
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


def _ingest_videos(
    videos: list[dict[str, Any]],
    skip_indexed: bool = True,
    label: str = "Processing videos",
) -> None:
    """Shared ingest logic: transcribe, embed, store a list of videos."""
    model: whisper.Whisper | None = None
    if not videos:
        console.print("[yellow]No videos to process.[/yellow]")
        return

    client = get_client()
    ensure_collection(client)

    # Pre-filter already-indexed videos to avoid loading Whisper unnecessarily
    if skip_indexed:
        to_process = [v for v in videos if not video_already_indexed(v["video_id"], client)]
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
        model = whisper.load_model(WHISPER_MODEL, device=device)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(label, total=len(to_process))

            for video in to_process:
                vid = video["video_id"]
                title = video["title"]
                url = video["url"]

                progress.update(task, description=f"[cyan]{title[:60]}[/cyan]")

                try:
                    # 1. Download audio + transcribe
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
                    console.print(f"  [red]✗ {title[:60]} — {e}[/red]")

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
    """Incremental ingest: fetch new videos from the channel and only process those."""
    _all, new_only = run_fetch_incremental(
        skip_uefn=skip_uefn,
        skip_automotive=skip_automotive,
        skip_archvis=skip_archvis,
        include_streams=include_streams,
    )

    if not new_only:
        return

    _ingest_videos(new_only, skip_indexed=True, label="Processing new videos")
