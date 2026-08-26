"""Thin, dependency-free wrappers around the ffmpeg/ffprobe binaries."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


class MediaError(RuntimeError):
    """ffmpeg/ffprobe exited non-zero, or the binary is missing."""


def _binary(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise MediaError(
            f"`{name}` not found on PATH. Install it with `brew install ffmpeg`."
        )
    return path


def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def run(args: list[str], *, timeout: int | None = None) -> str:
    """Run an ffmpeg-family command, raising MediaError with stderr on failure."""
    proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-12:]
        raise MediaError(
            f"{Path(args[0]).name} failed (exit {proc.returncode}):\n" + "\n".join(tail)
        )
    return proc.stdout


def probe(path: str | Path) -> dict[str, Any]:
    """Return duration, dimensions, fps and stream presence for a media file."""
    raw = run([
        _binary("ffprobe"), "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ])
    data = json.loads(raw)
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    duration = 0.0
    for candidate in (data.get("format", {}).get("duration"),
                      (video or {}).get("duration"),
                      (audio or {}).get("duration")):
        try:
            duration = float(candidate)
            break
        except (TypeError, ValueError):
            continue

    fps = 30.0
    if video and video.get("avg_frame_rate", "0/0") != "0/0":
        try:
            num, den = video["avg_frame_rate"].split("/")
            if float(den) != 0:
                fps = float(num) / float(den)
        except (ValueError, ZeroDivisionError):
            pass

    return {
        "duration": duration,
        "has_video": video is not None,
        "has_audio": audio is not None,
        "width": int(video["width"]) if video and video.get("width") else 0,
        "height": int(video["height"]) if video and video.get("height") else 0,
        "fps": round(fps, 3),
        "video_codec": (video or {}).get("codec_name"),
        "audio_codec": (audio or {}).get("codec_name"),
        "size_bytes": int(data.get("format", {}).get("size") or 0),
    }


def extract_audio(src: str | Path, dest: str | Path) -> Path:
    """Downmix to the 16 kHz mono PCM WAV that Whisper expects."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    run([
        _binary("ffmpeg"), "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src),
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        str(dest),
    ])
    return dest


def thumbnail(src: str | Path, dest: str | Path, at: float = 0.0, width: int = 540) -> Path | None:
    """Grab a single frame as a poster image. Returns None for audio-only input."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        run([
            _binary("ffmpeg"), "-y", "-hide_banner", "-loglevel", "error",
            "-ss", f"{max(at, 0):.3f}", "-i", str(src),
            "-frames:v", "1", "-vf", f"scale={width}:-2",
            str(dest),
        ])
    except MediaError:
        return None
    return dest if dest.exists() else None
