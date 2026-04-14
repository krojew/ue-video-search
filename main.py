#!/usr/bin/env python3
"""CLI entrypoint for the Unreal Engine Video Search tool."""

from __future__ import annotations

import os

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group()
def cli() -> None:
    """Unreal Engine YouTube Video Search — fetch, transcribe, embed, search."""


@cli.command()
@click.option("--refresh", is_flag=True, help="Re-fetch video list from YouTube instead of using cache.")
@click.option("--skip-uefn/--no-skip-uefn", default=True, help="Skip videos with UEFN or Fortnite in title (default: True).")
@click.option("--skip-automotive/--no-skip-automotive", default=True, help="Skip videos with automotive in title (default: True).")
@click.option("--skip-archvis/--no-skip-archvis", default=True, help="Skip videos with archvis in title (default: True).")
def fetch(
    refresh: bool,
    skip_uefn: bool,
    skip_automotive: bool,
    skip_archvis: bool,
) -> None:
    """Fetch the video list from the Unreal Engine YouTube channel."""
    from src.pipeline import run_fetch

    videos = run_fetch(
        use_cached=not refresh,
        skip_uefn=skip_uefn,
        skip_automotive=skip_automotive,
        skip_archvis=skip_archvis,
    )

    table = Table(title=f"Videos ({len(videos)})")
    table.add_column("#", style="dim", width=4)
    table.add_column("Title", max_width=60)
    table.add_column("Duration")
    table.add_column("Published")

    for i, v in enumerate(videos, 1):
        table.add_row(str(i), v["title"][:60], v["duration_text"], v["published_text"])

    console.print(table)


@cli.command()
@click.option("--refresh", is_flag=True, help="Re-fetch video list before ingesting.")
@click.option("--reindex", is_flag=True, help="Re-index videos even if already in Qdrant.")
@click.option("--update", is_flag=True, help="Incremental mode: fetch new videos only and ingest them.")
@click.option("--skip-uefn/--no-skip-uefn", default=True, help="Skip videos with UEFN or Fortnite in title (default: True).")
@click.option("--skip-automotive/--no-skip-automotive", default=True, help="Skip videos with automotive in title (default: True).")
@click.option("--skip-archvis/--no-skip-archvis", default=True, help="Skip videos with archvis in title (default: True).")
def ingest(
    refresh: bool,
    reindex: bool,
    update: bool,
    skip_uefn: bool,
    skip_automotive: bool,
    skip_archvis: bool,
) -> None:
    """Run the full ingest pipeline: download audio, transcribe, embed, store.

    Use --update to only fetch and process videos that are new since the last run.
    """
    if update:
        from src.pipeline import run_ingest_new_only

        run_ingest_new_only(
            skip_uefn=skip_uefn,
            skip_automotive=skip_automotive,
            skip_archvis=skip_archvis,
        )
        return

    from src.pipeline import run_fetch, run_ingest

    videos = run_fetch(
        use_cached=not refresh,
        skip_uefn=skip_uefn,
        skip_automotive=skip_automotive,
        skip_archvis=skip_archvis,
    )
    run_ingest(
        videos,
        skip_indexed=not reindex,
        skip_uefn=skip_uefn,
        skip_automotive=skip_automotive,
        skip_archvis=skip_archvis,
    )

@cli.command()
@click.argument("query")
@click.option("--top-k", default=10, help="Number of results to return.")
def search(query: str, top_k: int) -> None:
    """Search indexed videos for a topic."""
    from src.search import search_videos

    try:
        results = search_videos(query, top_k=top_k)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        return
    except ConnectionError as e:
        console.print(f"[red]Connection Error:[/red] {e}")
        return
    except RuntimeError as e:
        console.print(f"[red]Search Error:[/red] {e}")
        return
    except Exception as e:
        console.print(f"[red]Unexpected Error:[/red] {e}")
        return

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    console.print(f"\n[bold]Search results for:[/bold] [cyan]{query}[/cyan]\n")

    # Group results by video
    seen_videos: dict[str, list] = {}
    for r in results:
        vid_url = r["video_url"]
        if vid_url not in seen_videos:
            seen_videos[vid_url] = []
        seen_videos[vid_url].append(r)

    for video_url, hits in seen_videos.items():
        title = hits[0]["video_title"]
        console.print(f"[bold green]▶ {title}[/bold green]")
        console.print(f"  [dim]{video_url}[/dim]")

        for hit in hits:
            score_pct = f"{hit['score']:.1%}"
            console.print(
                f"  [yellow]{hit['time_range']}[/yellow]  "
                f"[dim](score: {score_pct})[/dim]"
            )
            console.print(f"    [link={hit['timestamped_url']}]{hit['timestamped_url']}[/link]")
            # Show a short excerpt
            excerpt = hit["excerpt"][:200].replace("\n", " ")
            console.print(f"    [dim]{excerpt}...[/dim]")
        console.print()


@cli.command()
@click.option("--host", default="0.0.0.0", help="Bind address.")
@click.option("--port", default=8000, help="Port to listen on.")
def serve(host: str, port: int) -> None:
    """Start the web application."""
    import uvicorn

    console.print(f"[bold]Starting web server on {host}:{port}[/bold]")
    uvicorn.run("src.webapp:app", host=host, port=port, reload=False)


@cli.command()
def interactive() -> None:
    """Interactive search mode — keep searching without restarting."""
    from src.search import search_videos

    console.print("[bold]Interactive search mode[/bold] (type 'quit' to exit)\n")

    while True:
        try:
            query = console.input("[bold cyan]Search:[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not query or query.lower() in ("quit", "exit", "q"):
            break

        results = search_videos(query, top_k=10)

        if not results:
            console.print("[yellow]No results found.[/yellow]\n")
            continue

        seen_videos: dict[str, list] = {}
        for r in results:
            vid_url = r["video_url"]
            if vid_url not in seen_videos:
                seen_videos[vid_url] = []
            seen_videos[vid_url].append(r)

        for video_url, hits in seen_videos.items():
            title = hits[0]["video_title"]
            console.print(f"\n[bold green]▶ {title}[/bold green]")
            console.print(f"  [dim]{video_url}[/dim]")

            for hit in hits:
                score_pct = f"{hit['score']:.1%}"
                console.print(
                    f"  [yellow]{hit['time_range']}[/yellow]  "
                    f"[dim](score: {score_pct})[/dim]"
                )
                console.print(f"    [link={hit['timestamped_url']}]{hit['timestamped_url']}[/link]")
                excerpt = hit["excerpt"][:200].replace("\n", " ")
                console.print(f"    [dim]{excerpt}...[/dim]")

        console.print()

    console.print("[dim]Goodbye.[/dim]")


if __name__ == "__main__":
    cli()
