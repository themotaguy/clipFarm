"""Stage 6 — cut each scored span into a vertical clip with burned-in captions.

Layout is the standard short-form treatment: the source frame is scaled to fit
the 9:16 canvas and centred over a blurred, darkened copy of itself that fills
the rest. Word-timed ASS captions are burned on top.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any, Callable, Sequence

import config
from app.core import captions as captions_mod
from app.core import media
from app.core import transcribe as transcribe_mod

ProgressFn = Callable[[float], None]

_TIME_RE = re.compile(r"^out_time_(?:us|ms)=(\d+)$")


def _run_ffmpeg(
    args: list[str],
    *,
    duration: float,
    cwd: Path | None = None,
    on_progress: ProgressFn | None = None,
) -> None:
    """Run ffmpeg, translating its -progress stream into a 0..1 fraction."""
    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(cwd) if cwd else None,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        match = _TIME_RE.match(line.strip())
        if match and on_progress and duration > 0:
            # out_time_us is microseconds, out_time_ms is (confusingly) also micro.
            seconds = int(match.group(1)) / 1_000_000
            on_progress(max(0.0, min(1.0, seconds / duration)))
    proc.wait()
    if proc.returncode != 0:
        stderr = (proc.stderr.read() if proc.stderr else "") or ""
        tail = stderr.strip().splitlines()[-12:]
        raise media.MediaError("ffmpeg failed while rendering:\n" + "\n".join(tail))


def _video_filters(width: int, height: int, subtitle_file: str | None) -> str:
    chain = (
        f"[0:v]split=2[bg][fg];"
        f"[bg]scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},gblur=sigma=20,eq=brightness=-0.12:saturation=1.15[bgb];"
        f"[fg]scale={width}:{height}:force_original_aspect_ratio=decrease[fgs];"
        f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2:format=auto,format=yuv420p[vs]"
    )
    if subtitle_file:
        return chain + f";[vs]subtitles={subtitle_file}[vout]"
    return chain + ";[vs]null[vout]"


def _audio_only_filters(width: int, height: int, subtitle_file: str | None) -> str:
    chain = (
        f"[1:a]showwaves=s={width}x{height // 4}:mode=cline:rate=30:"
        f"colors=0x38bdf8|0x818cf8[wave];"
        f"[0:v][wave]overlay=0:(H-h)/2:shortest=1,format=yuv420p[vs]"
    )
    if subtitle_file:
        return chain + f";[vs]subtitles={subtitle_file}[vout]"
    return chain + ";[vs]null[vout]"


def render_clip(
    source: str | Path,
    out_path: str | Path,
    start: float,
    end: float,
    segments: Sequence[dict[str, Any]],
    *,
    has_video: bool = True,
    burn_captions: bool | None = None,
    width: int | None = None,
    height: int | None = None,
    on_progress: ProgressFn | None = None,
) -> dict[str, Any]:
    """Cut and render one vertical clip. Returns paths and probe info."""
    width = width or config.RENDER_WIDTH
    height = height or config.RENDER_HEIGHT
    burn = config.BURN_CAPTIONS if burn_captions is None else burn_captions

    out_path = Path(out_path)
    work_dir = out_path.parent
    work_dir.mkdir(parents=True, exist_ok=True)
    duration = max(0.1, end - start)

    # Written next to the output so the filtergraph can reference a bare filename
    # (ffmpeg's subtitles= filter needs painful escaping for absolute paths).
    ass_name = f"{out_path.stem}.ass"
    ass_path = work_dir / ass_name
    captions_mod.write_ass(
        ass_path,
        captions_mod.build_ass(segments, start, end, width=width, height=height),
    )

    # Sidecar SRT, handy for re-uploading elsewhere.
    srt_path = work_dir / f"{out_path.stem}.srt"
    in_range = [s for s in segments if s["end"] > start and s["start"] < end]
    srt_path.write_text(transcribe_mod.to_srt(in_range, offset=start), encoding="utf-8")

    subtitle_arg = ass_name if burn else None

    if has_video:
        args = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-nostats",
            "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", str(source),
            "-filter_complex", _video_filters(width, height, subtitle_arg),
            "-map", "[vout]", "-map", "0:a?",
        ]
    else:
        args = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-nostats",
            "-f", "lavfi", "-t", f"{duration:.3f}",
            "-i", f"color=c=0x0b0f14:s={width}x{height}:r=30",
            "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", str(source),
            "-filter_complex", _audio_only_filters(width, height, subtitle_arg),
            "-map", "[vout]", "-map", "1:a",
        ]

    args += [
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-profile:v", "high", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2",
        "-movflags", "+faststart",
        "-progress", "pipe:1",
        str(out_path),
    ]

    _run_ffmpeg(args, duration=duration, cwd=work_dir, on_progress=on_progress)

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise media.MediaError(f"render produced no output at {out_path}")

    poster = media.thumbnail(
        out_path, work_dir / f"{out_path.stem}.jpg", at=min(1.0, duration / 3), width=405
    )
    info = media.probe(out_path)

    return {
        "path": str(out_path),
        "filename": out_path.name,
        "thumbnail": str(poster) if poster else None,
        "subtitles_ass": str(ass_path),
        "subtitles_srt": str(srt_path),
        "size_bytes": out_path.stat().st_size,
        "rendered_duration": round(info["duration"], 3),
        "width": info["width"],
        "height": info["height"],
    }
