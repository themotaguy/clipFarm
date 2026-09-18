"""Thin, dependency-free wrappers around the ffmpeg/ffprobe binaries."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


class MediaError(RuntimeError):
    """ffmpeg/ffprobe exited non-zero, or the binary is missing."""


#: Keg-only / non-PATH locations worth checking for a libass-capable build.
#: Homebrew's plain `ffmpeg` formula omits libass; `ffmpeg-full` includes it.
_EXTRA_PREFIXES = (
    "/opt/homebrew/opt/ffmpeg-full/bin",
    "/usr/local/opt/ffmpeg-full/bin",
    "/opt/homebrew/bin",
    "/usr/local/bin",
)

_resolved: dict[str, str] = {}


def _override(name: str) -> str:
    import config

    return (config.FFMPEG_BINARY if name == "ffmpeg" else config.FFPROBE_BINARY) or ""


def _candidates(name: str) -> list[str]:
    found: list[str] = []
    on_path = shutil.which(name)
    if on_path:
        found.append(on_path)
    for prefix in _EXTRA_PREFIXES:
        candidate = Path(prefix) / name
        if candidate.is_file() and str(candidate) not in found:
            found.append(str(candidate))
    return found


def _binary(name: str) -> str:
    """Absolute path to an ffmpeg-family binary.

    An explicit override wins. Otherwise the one on PATH is used, unless it
    cannot burn captions and another installed build can — a plain Homebrew
    ffmpeg has no libass, so it would fail every caption render.
    """
    if name in _resolved:
        return _resolved[name]

    chosen = _override(name)
    if not chosen:
        options = _candidates(name)
        if not options:
            raise MediaError(
                f"`{name}` not found on PATH. Install it with `brew install ffmpeg`."
            )
        chosen = options[0]
        if name == "ffmpeg" and not _probe_filter(chosen, "subtitles"):
            better = next(
                (o for o in options[1:] if _probe_filter(o, "subtitles")), None
            )
            if better:
                chosen = better

    if not Path(chosen).is_file() and not shutil.which(chosen):
        raise MediaError(f"`{name}` binary not found at `{chosen}`.")

    _resolved[name] = chosen
    return chosen


def binary(name: str) -> str:
    """Public accessor for the resolved ffmpeg/ffprobe path."""
    return _binary(name)


def resolved_binaries() -> dict[str, str]:
    """Which ffmpeg/ffprobe this process will actually run (for diagnostics)."""
    out: dict[str, str] = {}
    for name in ("ffmpeg", "ffprobe"):
        try:
            out[name] = _binary(name)
        except MediaError:
            out[name] = ""
    return out


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


_filter_cache: dict[tuple[str, str], bool] = {}


def _probe_filter(binary: str, name: str) -> bool:
    """Whether `binary` exposes the named filter. Cached per (binary, filter)."""
    key = (binary, name)
    if key in _filter_cache:
        return _filter_cache[key]

    available = False
    try:
        out = subprocess.run(
            [binary, "-hide_banner", "-filters"],
            capture_output=True, text=True, timeout=15,
        ).stdout
        # Rows look like " T.. subtitles  V->V  Render text subtitles onto input."
        available = any(
            line.split()[1] == name
            for line in out.splitlines()
            if len(line.split()) >= 2
        )
    except (subprocess.SubprocessError, OSError):
        available = False

    _filter_cache[key] = available
    return available


def has_filter(name: str) -> bool:
    """Whether the ffmpeg we will actually run exposes the named filter.

    Builds vary a lot: Homebrew's plain `ffmpeg` formula ships without libass,
    so `subtitles` is simply absent there (`ffmpeg-full` includes it). Callers
    use this to degrade gracefully instead of dying inside a filtergraph.
    """
    try:
        return _probe_filter(_binary("ffmpeg"), name)
    except MediaError:
        return False
