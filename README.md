# Unreal Engine YouTube Video Search — v2.0

Fetches videos from the [Unreal Engine YouTube channel](https://www.youtube.com/unrealengine), transcribes them with faster-whisper, generates embeddings via Ollama, stores them in Qdrant, and provides semantic search with timestamped links.

## Features

- **YouTube Integration**: Automatically fetches video metadata from Unreal Engine's YouTube channel
- **Smart Filtering**: Configurable content filters to skip UEFN/Fortnite, automotive, and archvis videos; optional inclusion of live streams
- **GPU-Accelerated Transcription**: Uses faster-whisper (CTranslate2) with float16 on CUDA and int8 on CPU; quantization is selected automatically at load time
- **Pipelined Ingest**: The next video's audio is downloaded in the background while the current one transcribes, so the GPU rarely waits on the network
- **Cached Opus Audio**: Audio is stored as opus (the codec YouTube already serves), avoiding a 16 kHz WAV resample and shrinking the cache ~10×; Whisper resamples internally on load
- **Windowed Chunking**: 2-minute overlapping windows preserve topical context for semantic search, with native Whisper timestamps for accurate jump-to-moment links
- **Title-Anchored Embeddings**: Each chunk is embedded with its video title prepended so retrieval reflects the video's actual subject, not just incidental keyword mentions
- **Asymmetric Retrieval**: Query-side instruction prefix (Qwen3-style by default; configurable for BGE/E5/etc.) so queries and documents share an embedding space
- **Vector Search**: Semantic search using Qdrant vector database and Ollama embeddings, with title-keyword reranking, adjacent-chunk merging, and per-video diversity caps
- **Web Interface**: Modern web UI for searching and managing video ingestion
- **Docker Support**: Complete containerized setup with GPU support
- **CLI Tools**: Command-line interface for all operations

## Upgrading from v1.x

v2.0 changes how transcripts are chunked, what gets embedded, and how queries are formatted before retrieval. None of the v2.0 fixes will improve search results on a v1.x index — re-ingest is required.

What changed at the data layer:

- Transcripts are now produced as 2-minute overlapping windows instead of single-sentence segments. Old cached transcripts in `data/transcripts/*.json` are in the v1 sentence format; they short-circuit transcription if present, so they must be deleted to pick up the new windowing, the language pin, the UE-jargon glossary, and the upgraded default Whisper model (`base` → `small`).
- Document embeddings now include the video title (`"{title}\n\n{chunk}"`). v1 embeddings were transcript-only and won't benefit from title anchoring.
- Qdrant point IDs are derived from `(video_id, start_seconds)`. Because v2 chunks have different start times than v1 sentence chunks, re-ingesting on top of a v1 collection leaves the old chunks in place — the collection should be dropped first.
- YouTube titles are now read from every `runs[]` entry, so titles previously truncated mid-string will only be fixed for re-fetched videos.
- Audio downloads are now cached as `.opus` (the codec YouTube already serves) rather than 16 kHz `.wav`. Any `data/audio/*.wav` files left from v1.x are orphaned — the new code only looks for `.opus`, so the old files take disk space but are never read.
- The transcription backend has switched from `openai-whisper` to `faster-whisper` (CTranslate2). The cached transcript JSON format is unchanged, so this is purely a speed/quantization improvement — but `pip install -r requirements.txt` is required to pick up the new dependency.

Migration steps:

```powershell
# 1. Drop the Qdrant collection (old v1 chunks have different IDs and would linger)
Invoke-RestMethod -Method Delete http://localhost:6333/collections/ue_videos

# 2. Delete cached transcripts so the new Whisper settings re-run, and clear
#    any orphaned .wav files from the v1 audio cache (new code uses .opus)
Remove-Item -Recurse -Force data/transcripts
Remove-Item -Force data/audio/*.wav -ErrorAction SilentlyContinue

# 3. Refresh Python dependencies (faster-whisper replaces openai-whisper)
pip install -r requirements.txt

# 4. Re-fetch so titles are captured with the multi-run fix and filters re-applied
python main.py fetch --refresh

# 5. Re-ingest from scratch
python main.py ingest
```

The bundled snapshot in `snapshots/` is also a v1 artifact and will not reflect the v2.0 improvements — regenerate it after re-ingest if you redistribute it.

## Prerequisites

- **Docker & Docker Compose** — for Qdrant and Ollama services
- **Python 3.10+** — for running the application
- **ffmpeg** — required by Whisper and yt-dlp
- **NVIDIA GPU** (optional) — for accelerated transcription

## Quick Start with Docker

Visit [Docker Hub](https://hub.docker.com/r/krojew/ue-video-search) or:

```bash
# 1. Clone the repository
git clone <repository-url>
cd ue-video-search

# 2. Start all services (Qdrant, Ollama, and the app)
docker compose up --build

# 3. Pull the embedding model
docker compose exec ollama ollama pull qwen3-embedding

# 4. Open your browser to http://localhost:8000
```

## Manual Setup

```bash
# 1. Install system dependencies
# Ubuntu/Debian:
sudo apt-get update && sudo apt-get install -y ffmpeg

# macOS:
brew install ffmpeg

# 2. Start Qdrant and Ollama
docker compose up -d qdrant ollama

# 3. Pull the embedding model
docker compose exec ollama ollama pull qwen3-embedding

# 4. Install Python dependencies
pip install -r requirements.txt

# 5. Start the web application
python main.py serve
```

## Configuration

All settings can be configured via environment variables. Create a `.env` file or export them in your shell:

| Variable | Default | Description |
|---|---|---|
| `DATA_DIR` | `./data` | Directory for storing audio files, transcripts, and video metadata |
| `CHANNEL_URL` | `https://www.youtube.com/unrealengine` | YouTube channel URL to fetch videos from |
| `MAX_AGE_YEARS` | `3` | Only fetch videos from the last N years |
| `MIN_DURATION_SECONDS` | `900` | Minimum video duration in seconds (15 minutes) |
| `WHISPER_MODEL` | `small` | Whisper model size (`tiny`/`base`/`small`/`medium`/`large`) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `EMBEDDING_MODEL` | `qwen3-embedding:0.6b` | Ollama embedding model name |
| `EMBEDDING_DIM` | `1024` | Embedding vector dimensions (must match the chosen model) |
| `EMBEDDING_QUERY_INSTRUCTION` | *(UE-specific default)* | Task instruction injected into the query side of asymmetric retrieval models |
| `EMBEDDING_QUERY_TEMPLATE` | `Instruct: {instruction}\nQuery: {query}` | Format used to wrap queries. Receives `{instruction}` and `{query}`. Set to `{query}` to disable wrapping for models that don't use a query template |
| `QDRANT_HOST` | `localhost` | Qdrant server hostname |
| `QDRANT_PORT` | `6333` | Qdrant server port |
| `COLLECTION_NAME` | `ue_videos` | Qdrant collection name |
| `CHUNK_DURATION_SECONDS` | `120` | Target duration for transcript segments (seconds) |
| `CHUNK_OVERLAP_SECONDS` | `15` | Overlap between transcript segments (seconds) |

## Usage

### Command Line Interface

#### Fetch video list
```bash
# Fetch videos with default filters (skip UEFN, automotive, archvis)
python main.py fetch

# Re-fetch from YouTube (ignore cache)
python main.py fetch --refresh

# Fetch all videos (no filters)
python main.py fetch --no-skip-uefn --no-skip-automotive --no-skip-archvis
```

#### Ingest videos (download, transcribe, embed)
```bash
# Process all fetched videos
python main.py ingest

# Incremental mode: only process new videos
python main.py ingest --update

# Re-process already indexed videos
python main.py ingest --reindex

# Control content filtering during ingest
python main.py ingest --no-skip-automotive  # Include automotive videos
python main.py ingest --no-include-streams  # Exclude live streams
```

#### Search videos
```bash
# Search for content
python main.py search "nanite virtual geometry"
python main.py search "blueprint networking" --top-k 5
```

#### Purge stale videos
```bash
# Remove videos from the index that are no longer in the cached list
# (e.g. after tightening filters or shrinking MAX_AGE_YEARS)
python main.py purge
```

#### Interactive search
```bash
python main.py interactive
```

#### Start web server
```bash
python main.py serve --host 0.0.0.0 --port 8000
```

### Web Interface

The web interface provides:

- **Search**: Semantic search across all indexed videos with timestamped results
- **Ingest Management**: Start full or incremental ingestion with content filtering options
- **Real-time Progress**: Live updates during video processing
- **Statistics**: Overview of indexed videos and system status

Access it at `http://localhost:8000` after starting the server.

## Architecture

```
YouTube Channel
    │
    ▼
┌───────────┐     ┌───────────┐     ┌──────────┐     ┌────────────┐
│ scrapetube│───▶│  yt-dlp    │───▶│ Whisper  │───▶│ Window     │
│ (listing) │     │ (opus)    │     │ (STT)    │     │ Splitter   │
└───────────┘     └───────────┘     └──────────┘     └───┬────────┘
                                                         │
                                                         ▼
                                                    ┌───────────┐
                                                    │  Ollama   │
                                                    │(embedding)│
                                                    └─────┬─────┘
                                                          │
                                                          ▼
                                                    ┌───────────┐
                                                    │  Qdrant   │
                                                    │(vector DB)│
                                                    └─────┬─────┘
                                                          │
                                            search query  ▼
                                                    ┌───────────┐
                                                    │  Results  │
                                                    │ + links   │
                                                    └───────────┘
```

### Data Flow

1. **Fetch**: Use scrapetube to get video metadata from YouTube channel (yt-dlp is the fallback when scrapetube is blocked)
2. **Filter**: Apply content filters (UEFN/Fortnite, automotive, archvis) using word-boundary regex
3. **Download**: yt-dlp extracts the original opus audio stream (no resample); the next video is prefetched in the background while the current one transcribes
4. **Transcribe**: faster-whisper converts audio to text with timestamps, pinned to English with a UE-jargon initial prompt
5. **Segment**: Slide a 2-minute window over Whisper segments with 15s of overlap, preserving native timestamps
6. **Embed**: Prepend the video title to each chunk, then batch through Ollama for vector embeddings
7. **Store**: Save embeddings and metadata in Qdrant; point IDs are derived from `(video_id, start_seconds)` so re-ingest is idempotent
8. **Search**: Wrap the query with a model-specific instruction prefix, retrieve from Qdrant, then rerank with title-keyword boosting, adjacent-chunk merging, and per-video diversity caps

## Docker Deployment

### Build and run
```bash
# Build the image
docker build -t ue-video-search .

# Run with GPU support
docker run --rm -p 8000:8000 --gpus all \
  -e QDRANT_HOST=qdrant \
  -e OLLAMA_BASE_URL=http://ollama:11434 \
  --name ue-video-search ue-video-search

# Or use docker-compose for full stack
docker compose up --build
```

### GPU Requirements
- NVIDIA GPU with CUDA 11.8+ or 12.1+
- NVIDIA Container Toolkit installed
- `--gpus all` flag for GPU access

## Troubleshooting

### Common Issues

**"Connection refused" errors**
- Ensure Qdrant and Ollama containers are running
- Check network connectivity between containers

**GPU not detected**
- Verify NVIDIA drivers and CUDA installation
- Use `nvidia-smi` to check GPU status
- Ensure `--gpus all` flag is used with Docker

**Out of memory during transcription**
- Use smaller Whisper model (`tiny`, `base`)
- Process fewer videos at once
- Ensure adequate RAM (8GB+ recommended)

**No videos found**
- Check internet connectivity
- Verify YouTube channel URL is accessible
- Adjust `MAX_AGE_YEARS` if needed

### Logs and Debugging

```bash
# View container logs
docker compose logs -f

# Check Qdrant status
curl http://localhost:6333/health

# Check Ollama status
curl http://localhost:11434/api/tags
```

## Development

### Project Structure
```
├── src/
│   ├── config.py          # Configuration management
│   ├── fetcher.py         # YouTube video fetching
│   ├── transcriber.py     # Audio download and transcription
│   ├── embeddings.py      # Text embedding generation
│   ├── vectordb.py        # Qdrant vector database operations
│   ├── search.py          # Semantic search functionality
│   ├── webapp.py          # FastAPI web application
│   ├── ingest_worker.py   # Background ingestion worker
│   └── pipeline.py        # High-level pipeline orchestration
├── static/
│   └── index.html         # Web interface
├── data/                  # Audio, transcripts, and metadata
├── docker-compose.yml     # Multi-service setup
├── Dockerfile            # Container definition
└── requirements.txt      # Python dependencies
```

### Adding New Features

1. **New filters**: Add filter logic in `fetcher.py` and CLI options in `main.py`
2. **Different models**: Update `WHISPER_MODEL` or `EMBEDDING_MODEL` in config
3. **Custom segmentation**: Modify the `_window_segments` helper in `transcriber.py`, or tune `CHUNK_DURATION_SECONDS` / `CHUNK_OVERLAP_SECONDS` in `config.py`
4. **Additional metadata**: Extend data structures in `vectordb.py`

