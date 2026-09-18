# clipFarm

Turn a long-form video into ranked, ready-to-post vertical clips — entirely on your own machine.

Feed it an upload or a URL. It transcribes the audio with Whisper, streams the transcript
into a ChromaDB vector index while transcription is still running, retrieves the moments most
likely to travel, has a local Ollama model grade each one and choose tighter in/out points,
then renders 9:16 MP4s with word-timed captions. No API keys, no cloud, no per-minute billing.

```
                 ┌──────────── transcribe (Whisper) ────────────┐
upload / URL ──► │  segment ──► segment ──► segment ──► ...     │
   ingest        └───────┬──────────────────────┬───────────────┘
  (yt-dlp +             │ each finished segment │
   ffmpeg)              ▼                       ▼
                  WindowBuilder            (decode keeps going)
                  ~45s windows,
                   50% overlap
                        │
                        ▼  queue
                 BackgroundIndexer ──► ChromaDB  (embeddings via Ollama)
                        │
                        ▼
                  8 probe queries ──► candidate moments
                        │
                        ▼
                  Ollama scores each: hook, emotion, clarity, shareability
                        │                and picks tighter in/out points
                        ▼
                  ffmpeg renders 9:16 + burned captions ──► clips/
```

The transcribe and index stages are deliberately overlapped. Whisper hands each finished
segment to the window builder on the decode thread; completed windows go onto a queue that a
background thread drains into Chroma. Embedding latency never blocks the decoder, so the
vector index is warm the moment transcription ends.

## Requirements

- **Python 3.11** — `ctranslate2` and `chroma-hnswlib` have no wheels for 3.13+
- **ffmpeg + ffprobe** on `PATH` — `brew install ffmpeg`
- **[Ollama](https://ollama.com)** running locally, with two models pulled

```bash
brew install ffmpeg
ollama pull llama3.1:8b        # scoring
ollama pull nomic-embed-text   # embeddings
```

### Captions and libass

Homebrew's plain `ffmpeg` formula ships **without** libass, so it has no `subtitles` filter.
Captions still work: clipFarm has two burn-in backends and picks one automatically.

| Backend  | When it is used                          | Notes                              |
| -------- | ---------------------------------------- | ---------------------------------- |
| `libass` | ffmpeg has the `subtitles` filter        | Preferred — real text shaping      |
| `pillow` | it does not                              | Captions drawn with Pillow, composited via `overlay`; works on any build |

If a libass-capable ffmpeg is installed anywhere it knows about — including Homebrew's
keg-only `ffmpeg-full` — clipFarm finds and uses it even when the one on `PATH` cannot burn
captions. Pin a specific binary with `FFMPEG_BINARY` if you would rather choose yourself.

```bash
brew install ffmpeg-full       # optional; adds libass (plus ~45 other libraries)
```

`/api/health` reports which backend is live under `checks.captions.backend`. The Pillow path
is a little slower (one image per caption phrase) and its text shaping is simpler, but it is
visually equivalent: same grouping, same per-word highlight, same placement.

## Setup

```bash
git clone <this repo> && cd clipFarm
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env           # optional; every value has a default
python run.py
```

Open <http://127.0.0.1:5001>. Check <http://127.0.0.1:5001/api/health> first — it verifies
ffmpeg, Ollama (including whether your models are actually pulled), and ChromaDB before you
waste time on a long video.

Whisper downloads its weights on first run (~500 MB for `small`).

## API

Everything the browser UI does is available over HTTP.

| Method   | Path                                     | Purpose                                              |
| -------- | ---------------------------------------- | ---------------------------------------------------- |
| `GET`    | `/`                                      | Web UI                                               |
| `GET`    | `/api/health`                            | Dependency + config check; run this first            |
| `POST`   | `/api/jobs`                              | Start a job — multipart `file`, or JSON `{"url":…}`   |
| `GET`    | `/api/jobs`                              | List jobs, newest first                              |
| `GET`    | `/api/jobs/<id>`                         | Job state, stats and clips                           |
| `GET`    | `/api/jobs/<id>/events`                  | **SSE** progress stream (`state`, `log`, `end`)      |
| `POST`   | `/api/jobs/<id>/cancel`                  | Cancel a running job                                 |
| `DELETE` | `/api/jobs/<id>`                         | Delete the job, its media, clips and vectors         |
| `GET`    | `/api/jobs/<id>/transcript`              | Full transcript; `?format=srt` for a subtitle file   |
| `POST`   | `/api/jobs/<id>/search`                  | Semantic search over the job's transcript            |
| `GET`    | `/api/jobs/<id>/poster`                  | Source poster frame                                  |
| `GET`    | `/api/jobs/<id>/clips/<clip>/file`       | Clip MP4 (supports `Range`; `?download=1` to save)   |
| `GET`    | `/api/jobs/<id>/clips/<clip>/thumbnail`  | Clip thumbnail JPEG                                  |
| `GET`    | `/api/jobs/<id>/clips/<clip>/srt`        | Clip subtitles, rebased to the clip start            |

```bash
# Start a job from a file and follow it to completion.
JOB=$(curl -s -F "file=@talk.mp4" localhost:5001/api/jobs | jq -r .id)
curl -N localhost:5001/api/jobs/$JOB/events

# Ask the transcript a question once the job is indexed.
curl -s -X POST localhost:5001/api/jobs/$JOB/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"how do I make model serving cheaper?","k":3}' | jq
```

Completed jobs are written to `data/jobs/<id>.json` and reloaded at startup, so finished
clips survive a server restart.

## How a clip gets picked

1. **Retrieve.** Eight fixed probe queries describe *why* clips get shared — a counterintuitive
   claim, a shocking number, a quotable one-liner, and so on. Each runs against the job's
   Chroma collection. A window matched by several different probes scores higher than one that
   got lucky on a single narrow probe, and near-duplicate spans are suppressed before any LLM
   time is spent on them.
2. **Score.** Each surviving candidate goes to Ollama with its numbered transcript lines. The
   model returns a title, hook, tags, four sub-scores and an overall virality score — and picks
   the line to start and end on, which is then forced back inside the duration limits.
   `final_score = 0.75 × virality + 25 × retrieval_score`.
3. **Finish the thought.** A grammatical full stop is not the end of a point — "So there are
   three moves." is a complete sentence and a terrible place to stop. The clip runs on to the
   next *topic* boundary when one is within reach, located from discourse markers ("So,",
   "Now,", "Here's the thing"), from how long the speaker pauses (the median gap at a sentence
   end is ~0.26s; a change of subject runs to half a second or more), and optionally from
   embedding drift between neighbouring passages.
4. **Place the cuts.** The boundaries the model returns land on word edges, which sounds
   severed — the last word's decay gets chopped and there is no breath. So each edge is moved
   onto real silence: nearby pauses are scored, sentence breaks strongly preferred, and
   reaching forward to finish a sentence is treated as much cheaper than cutting a second
   early. No cut is ever left inside a word, including when a length cap or a budget forces
   one to move.
5. **Stitch, if the clip needs it.** A clip opening "The third move is simply not calling the
   model" makes no sense alone — nothing in it says what the first two were. When the opening
   refers to something it never explains, an earlier passage that introduces the referent is
   cut and played first, and the two spans are concatenated into one clip. The model is asked
   first; because a local 8B almost always insists the clip is self-contained, there is also a
   deterministic detector for enumerations ("the second option"), demonstratives ("that
   number"), dangling pronouns and explicit back-references ("as I said").
6. **Spread the lengths.** Left alone, every clip comes back about the same length. Each of
   the `MAX_CLIPS` slots is instead given its own slice of the allowed range — geometrically
   spaced, so the short end is finer — and the best-scoring clip in each slice is kept. A
   bucket may be overshot by 15% when that is what it takes to end cleanly; hitting a length
   target exactly is not worth ending mid-sentence for.
7. **Render.** The chosen spans are cut to 9:16 with word-timed captions on top. A landscape
   source is *cropped* to fill the frame rather than letterboxed over a blur, and the crop
   follows the subject: sampled frames are scored per column for detail and motion, and the
   crop window is placed over the peak (see `RENDER_FILL`).

If Ollama returns unparseable JSON, it retries once with a blunter instruction, then falls
back to a retrieval-only score so one bad response can't take down the run.

The main span is confined to the retrieved moment. The scorer is shown extra transcript
either side so it has somewhere to draw setup from, but left unconstrained it wanders off to
whatever it finds most quotable in that window — which discards retrieval's judgement and
collapses several candidates onto the same passage.

## Configuration

All optional — see `.env.example`. The defaults are what the numbers below were measured with.

| Variable                                   | Default             | Notes                                          |
| ------------------------------------------ | ------------------- | ---------------------------------------------- |
| `WHISPER_MODEL`                            | `small`             | `tiny`…`large-v3`, or `distil-large-v3`        |
| `WHISPER_COMPUTE_TYPE` / `WHISPER_DEVICE`  | `int8` / `cpu`      | `float16` + `cuda` if you have an NVIDIA GPU   |
| `WHISPER_LANGUAGE`                         | auto-detect         | Set e.g. `en` to skip detection                |
| `OLLAMA_MODEL`                             | `llama3.1:8b`       | Any chat model that can emit JSON              |
| `OLLAMA_EMBED_MODEL`                       | `nomic-embed-text`  | Embedding model                                |
| `CLIP_MIN_SECONDS` / `CLIP_MAX_SECONDS`    | `15` / `180`        | Hard bounds on clip length                     |
| `DURATION_SPREAD`                          | `true`              | Spread clip lengths across that range          |
| `TOPIC_COMPLETION`                         | `true`              | Run on to the end of the thought               |
| `TOPIC_EXTEND_SECONDS`                     | `45`                | Furthest a clip may run on to finish a point   |
| `TOPIC_SHIFT_THRESHOLD`                    | `0.62`              | Similarity below which a topic has changed     |
| `RENDER_FILL`                              | `crop`              | `crop` / `blur` / `auto` for landscape sources |
| `FRAMING_SAMPLES`                          | `12`                | Frames sampled to locate the subject           |
| `WINDOW_MAX_SECONDS`                       | `90`                | Retrieval window cap, independent of clip cap  |
| `CLIP_TARGET_SECONDS`                      | `45`                | Window size for retrieval and clipping         |
| `MAX_CLIPS`                                | `5`                 | Clips rendered per job                         |
| `SCORE_CANDIDATES`                         | `14`                | Candidates sent to the LLM                     |
| `EMBED_BATCH_SIZE`                         | `16`                | Windows per Chroma write                       |
| `RENDER_WIDTH` / `RENDER_HEIGHT`           | `1080` / `1920`     | Output canvas                                  |
| `BURN_CAPTIONS`                            | `true`              | Master switch for burned-in captions           |
| `CAPTION_RENDERER`                         | `auto`              | `auto` / `libass` / `pillow` / `none`          |
| `CAPTION_FONT`                             | `Arial Black`       | Caption typeface                               |
| `FFMPEG_BINARY` / `FFPROBE_BINARY`         | auto-detect         | Pin a build; blank prefers one with libass     |
| `CLIP_LEAD_IN_SECONDS` / `CLIP_TAIL_SECONDS` | `0.25` / `0.6`    | Breathing room at each end of a cut            |
| `CLIP_SNAP_EXTEND`                         | `10.0`              | How far a cut may reach *forward* for a pause  |
| `CLIP_SNAP_TRUNCATE`                       | `1.5`               | How far it may cut *back* (deliberately less)  |
| `CLIP_MIN_PAUSE`                           | `0.18`              | Shortest gap that counts as silence            |
| `STITCH_SETUP`                             | `true`              | Allow a setup span before the payoff           |
| `SETUP_MAX_SECONDS`                        | `20`                | Longest setup span                             |
| `SETUP_CONTEXT_SECONDS`                    | `60`                | Transcript shown either side of a candidate    |
| `FLASK_HOST` / `FLASK_PORT`                | `127.0.0.1` / `5001`|                                                |
| `MAX_UPLOAD_MB`                            | `2048`              | Upload ceiling                                 |

## Performance

A 4m12s talking-head video, all six stages, on an Apple Silicon Mac with `WHISPER_MODEL=small`
on CPU and `llama3.1:8b` via Ollama:

| Stage                    | Result                                       |
| ------------------------ | -------------------------------------------- |
| Transcribe               | 77 segments, 797 words, language `en` p=0.997 |
| Index                    | 12 windows → 12 vectors in ChromaDB          |
| Retrieve                 | 8 candidates from 8 probe queries            |
| Score                    | 8 scored by `llama3.1:8b`, top score 85      |
| Render                   | 5 clips, 1080×1920, 5–9 MB each              |
| **Total**                | **~250 s wall clock** (≈1× realtime)         |

Transcription dominates on CPU; a GPU or `distil-large-v3` moves the needle most.

## Layout

```
app/
  api/routes.py       REST + SSE endpoints
  core/ingest.py      stage 1 — yt-dlp / upload → local media + 16 kHz wav
  core/transcribe.py  stage 2 — faster-whisper, streams segments to a callback
  core/chunker.py     stage 3 — sliding time windows, LangChain splitter as a guard
  core/vectorstore.py stage 3 — batched write-through buffer over ChromaDB
  core/scoring.py     stages 4 & 5 — probe retrieval, then Ollama grading
  core/boundaries.py  moving cut points onto natural pauses
  core/topics.py      where a thought actually finishes
  core/framing.py     locating the subject so a landscape crop keeps them in shot
  core/context.py     spotting clips that open on an unexplained reference
  core/timeline.py    source-time <-> clip-time mapping for stitched clips
  core/pngcaptions.py caption burn-in for ffmpeg builds without libass
  core/render.py      stage 6 — ffmpeg 9:16 composite
  core/captions.py    word-timed ASS subtitles
  core/pipeline.py    orchestration; overlaps transcribe with indexing
  core/media.py       ffmpeg/ffprobe wrappers and capability probes
  jobs.py             job registry, progress, pub/sub bus behind the SSE stream
config.py             environment-backed settings
tests/                pytest suite
```

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

293 tests, about fifteen seconds. The suite stubs Whisper and Ollama, so it needs neither; the
`media` and `render` tests do shell out to a real ffmpeg and skip themselves if it is absent.

## Troubleshooting

| Symptom                                         | Cause / fix                                                                 |
| ----------------------------------------------- | --------------------------------------------------------------------------- |
| `cannot reach http://127.0.0.1:11434`           | Ollama isn't running — `ollama serve`                                       |
| `missing model(s)`                              | `ollama pull llama3.1:8b` / `ollama pull nomic-embed-text`                   |
| Clips have no captions                          | Check `checks.captions` in `/api/health`; install Pillow or an ffmpeg with libass |
| Captions look plainer than expected             | You are on the `pillow` backend; install `ffmpeg-full` for libass shaping    |
| A clip is labelled "2 parts"                    | It was stitched to include its own setup — hover the badge for the spans     |
| A clip still ends mid-sentence                  | The nearest full stop was further than `CLIP_SNAP_EXTEND`; raise it          |
| A clip stops before the point is made           | No topic boundary within `TOPIC_EXTEND_SECONDS`; raise it or `CLIP_MAX_SECONDS` |
| The subject is cropped out of frame             | Set `RENDER_FILL=blur`, or `auto` to crop only when a subject is found       |
| All clips come back the same length             | `DURATION_SPREAD=false`, or `MAX_CLIPS=1` leaves only one bucket             |
| `Whisper produced no speech`                    | No audible dialogue, or the wrong `WHISPER_LANGUAGE`                        |
| `No candidate could be scored`                  | Ollama is up but the model can't emit JSON — try another model               |
| `no wheels` on install                          | You're on Python 3.13+; use 3.11                                            |
| Very slow transcription                         | Expected on CPU; try `WHISPER_MODEL=base`, or `float16` on a GPU             |
