"""Central configuration, loaded from the environment (see .env.example)."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))

UPLOAD_DIR = DATA_DIR / "uploads"
MEDIA_DIR = DATA_DIR / "media"
CLIP_DIR = DATA_DIR / "clips"
CHROMA_DIR = DATA_DIR / "chroma"
JOB_DIR = DATA_DIR / "jobs"
MODEL_DIR = DATA_DIR / "models"

for _d in (UPLOAD_DIR, MEDIA_DIR, CLIP_DIR, CHROMA_DIR, JOB_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def _flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# --- Whisper ---
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE") or None

# --- Ollama ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))

# --- clip shaping ---
CLIP_MIN_SECONDS = float(os.getenv("CLIP_MIN_SECONDS", "15"))
CLIP_MAX_SECONDS = float(os.getenv("CLIP_MAX_SECONDS", "180"))
CLIP_TARGET_SECONDS = float(os.getenv("CLIP_TARGET_SECONDS", "45"))
MAX_CLIPS = int(os.getenv("MAX_CLIPS", "5"))
SCORE_CANDIDATES = int(os.getenv("SCORE_CANDIDATES", "14"))

# Spread the selected clips across the allowed length range instead of letting
# them all cluster near the target: each slot gets its own duration bucket.
DURATION_SPREAD = _flag("DURATION_SPREAD", True)

# Retrieval windows stay short regardless of CLIP_MAX_SECONDS — a 180s window
# would be far too coarse to locate a moment with.
WINDOW_MAX_SECONDS = float(os.getenv("WINDOW_MAX_SECONDS", "90"))

# --- sponsor / ad segments ---
# Integrated ads make tempting clips and worthless ones: scripted, punchy and
# entirely off-topic. Detected reads are excluded from candidate selection.
AD_DETECTION = _flag("AD_DETECTION", True)
# A candidate with at least this fraction inside an ad is discarded.
AD_OVERLAP_THRESHOLD = float(os.getenv("AD_OVERLAP_THRESHOLD", "0.25"))
# How far apart two mentions can be and still count as one read.
AD_GAP_SECONDS = float(os.getenv("AD_GAP_SECONDS", "30"))
AD_MIN_SECONDS = float(os.getenv("AD_MIN_SECONDS", "6"))

# Channel housekeeping ("hit the subscribe button") is excluded alongside
# sponsor reads, and excised from the middle of a clip rather than truncating
# it — the surrounding halves are stitched back together.
FILLER_DETECTION = _flag("FILLER_DETECTION", True)

# --- narrative shape ---
# The scorer names the most scroll-stopping sentence but then routinely starts
# the clip 6-43s earlier, on preamble. These re-anchor the clip onto it.
HOOK_ANCHOR = _flag("HOOK_ANCHOR", True)
HOOK_RUN_UP_SECONDS = float(os.getenv("HOOK_RUN_UP_SECONDS", "1.0"))
# Capped so a badly matched hook cannot gut an otherwise good clip.
HOOK_TRIM_MAX_SECONDS = float(os.getenv("HOOK_TRIM_MAX_SECONDS", "15"))
# A near-verbatim hook quote can be trusted to skip a longer preamble; the
# tighter cap above applies when the hook was only matched fuzzily.
HOOK_TRIM_STRONG_SECONDS = float(os.getenv("HOOK_TRIM_STRONG_SECONDS", "45"))
HOOK_STRONG_MATCH = float(os.getenv("HOOK_STRONG_MATCH", "0.85"))
HOOK_MATCH_THRESHOLD = float(os.getenv("HOOK_MATCH_THRESHOLD", "0.6"))

# Never end on a buildup whose payoff lands just after the cut.
REVEAL_COMPLETION = _flag("REVEAL_COMPLETION", True)
REVEAL_WINDOW_SECONDS = float(os.getenv("REVEAL_WINDOW_SECONDS", "25"))

# A second pass over the finalists only, checking they open on the hook and
# contain the payoff. Advisory: its suggestions must clear the same
# deterministic gates before they are adopted.
CRITIC_PASS = _flag("CRITIC_PASS", True)

# --- topic completion ---
# A clip that stops on a grammatical full stop can still stop mid-argument.
# These control how far a clip may run on to finish the thought it started.
TOPIC_COMPLETION = _flag("TOPIC_COMPLETION", True)
TOPIC_EXTEND_SECONDS = float(os.getenv("TOPIC_EXTEND_SECONDS", "45"))
# Cosine similarity below this between neighbouring passages = topic shift.
TOPIC_SHIFT_THRESHOLD = float(os.getenv("TOPIC_SHIFT_THRESHOLD", "0.62"))

# Vector layer: chunks are flushed to Chroma in batches this size as the
# transcript streams in, so retrieval is warm before transcription finishes.
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "16"))

# --- cut points ---
# Boundaries handed over by the LLM land on word edges, which sounds severed.
# These control how far a cut may move to find real silence to land in.
CLIP_LEAD_IN_SECONDS = float(os.getenv("CLIP_LEAD_IN_SECONDS", "0.25"))
CLIP_TAIL_SECONDS = float(os.getenv("CLIP_TAIL_SECONDS", "0.6"))
# Asymmetric on purpose: reaching forward to let a sentence finish is cheap,
# cutting backwards throws away a moment the scorer chose.
# 10s measured best on real footage: 15/15 clips ended on a sentence versus
# 13/15 at 6s, for only ~1.6s of added length on average.
CLIP_SNAP_EXTEND = float(os.getenv("CLIP_SNAP_EXTEND", "10.0"))
CLIP_SNAP_TRUNCATE = float(os.getenv("CLIP_SNAP_TRUNCATE", "1.5"))
CLIP_MIN_PAUSE = float(os.getenv("CLIP_MIN_PAUSE", "0.18"))

# --- stitching ---
# A clip may prepend one earlier "setup" span when the payoff depends on
# context stated before it. The model is shown this much extra transcript
# either side of the candidate so it has something to choose from.
STITCH_SETUP = _flag("STITCH_SETUP", True)
SETUP_MAX_SECONDS = float(os.getenv("SETUP_MAX_SECONDS", "20"))
SETUP_CONTEXT_SECONDS = float(os.getenv("SETUP_CONTEXT_SECONDS", "60"))

# --- render ---
RENDER_WIDTH = int(os.getenv("RENDER_WIDTH", "1080"))
RENDER_HEIGHT = int(os.getenv("RENDER_HEIGHT", "1920"))

# How a landscape source is fitted to the vertical canvas.
#   crop - fill the frame by cropping, tracking the main subject (default)
#   blur - letterbox the whole frame over a blurred copy of itself
#   auto - crop when a subject can be located, else blur
RENDER_FILL = os.getenv("RENDER_FILL", "crop").strip().lower()
# Sample this many frames per span when locating the subject.
FRAMING_SAMPLES = int(os.getenv("FRAMING_SAMPLES", "12"))
# Re-frame on every cut instead of using one crop for the whole clip. A single
# position is a compromise across every shot: measured on a 95s talking-head
# clip with b-roll inserts, the best crop moved 7% of the frame width between
# shots, a quarter of the crop window.
FRAMING_PER_SHOT = _flag("FRAMING_PER_SHOT", True)
# Within a shot, follow whoever is speaking. Only bites on a locked-off shot
# holding two or more faces — an interview, a podcast wide — where the editor
# never cuts and per-shot framing has nothing to work with. Needs mediapipe
# and a one-off ~230 KB model download; falls back to per-shot framing if
# either is missing, or if the evidence does not clearly favour one face.
SPEAKER_TRACKING = _flag("SPEAKER_TRACKING", True)
BURN_CAPTIONS = _flag("BURN_CAPTIONS", True)
# auto   - libass if this ffmpeg has it, else the built-in Pillow renderer
# libass - require ffmpeg's subtitles filter
# pillow - always draw captions ourselves
# none   - equivalent to BURN_CAPTIONS=false
CAPTION_RENDERER = os.getenv("CAPTION_RENDERER", "auto").strip().lower()
CAPTION_FONT = os.getenv("CAPTION_FONT", "Arial Black")

# Point these at a specific build (e.g. Homebrew's keg-only `ffmpeg-full`,
# which unlike plain `ffmpeg` ships with libass). Blank = search PATH, and
# prefer a libass-capable build if the one on PATH cannot burn captions.
FFMPEG_BINARY = os.getenv("FFMPEG_BINARY", "").strip()
FFPROBE_BINARY = os.getenv("FFPROBE_BINARY", "").strip()

# --- server ---
FLASK_HOST = os.getenv("FLASK_HOST", "127.0.0.1")
FLASK_PORT = int(os.getenv("FLASK_PORT", "5001"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "2048"))

ALLOWED_EXTENSIONS = {
    ".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".flv",
    ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus",
}
