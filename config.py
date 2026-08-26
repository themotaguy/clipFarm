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

for _d in (UPLOAD_DIR, MEDIA_DIR, CLIP_DIR, CHROMA_DIR, JOB_DIR):
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
CLIP_MIN_SECONDS = float(os.getenv("CLIP_MIN_SECONDS", "18"))
CLIP_MAX_SECONDS = float(os.getenv("CLIP_MAX_SECONDS", "75"))
CLIP_TARGET_SECONDS = float(os.getenv("CLIP_TARGET_SECONDS", "45"))
MAX_CLIPS = int(os.getenv("MAX_CLIPS", "5"))
SCORE_CANDIDATES = int(os.getenv("SCORE_CANDIDATES", "14"))

# Vector layer: chunks are flushed to Chroma in batches this size as the
# transcript streams in, so retrieval is warm before transcription finishes.
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "16"))

# --- render ---
RENDER_WIDTH = int(os.getenv("RENDER_WIDTH", "1080"))
RENDER_HEIGHT = int(os.getenv("RENDER_HEIGHT", "1920"))
BURN_CAPTIONS = _flag("BURN_CAPTIONS", True)

# --- server ---
FLASK_HOST = os.getenv("FLASK_HOST", "127.0.0.1")
FLASK_PORT = int(os.getenv("FLASK_PORT", "5001"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "2048"))

ALLOWED_EXTENSIONS = {
    ".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".flv",
    ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus",
}
