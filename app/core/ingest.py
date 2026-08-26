"""Stage 1 — turn an upload or a URL into a local media file plus Whisper-ready audio."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import config
from app.core import media
from app.jobs import Job, StageReporter

_URL_RE = re.compile(r"^https?://", re.IGNORECASE)


def is_url(value: str) -> bool:
    return bool(_URL_RE.match(value.strip()))


def download(url: str, job_id: str, reporter: StageReporter) -> tuple[Path, dict[str, Any]]:
    """Fetch a remote video with yt-dlp, reporting download progress as we go."""
    import yt_dlp

    out_dir = config.MEDIA_DIR / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    def hook(d: dict[str, Any]) -> None:
        reporter.check_cancelled()
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            done = d.get("downloaded_bytes") or 0
            if total:
                # Downloading owns the first 70% of the ingest stage.
                reporter.progress(0.7 * (done / total),
                                  f"Downloading… {done / 1e6:.1f} / {total / 1e6:.1f} MB")
        elif d.get("status") == "finished":
            reporter.progress(0.7, "Download complete, remuxing…")

    opts = {
        "outtmpl": str(out_dir / "source.%(ext)s"),
        "format": "bv*[height<=1080]+ba/b[height<=1080]/bv*+ba/b",
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [hook],
        "retries": 3,
        "concurrent_fragment_downloads": 4,
    }

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)

    # yt-dlp may have merged into a different container than the template guessed.
    candidates = sorted(out_dir.glob("source.*"), key=lambda p: p.stat().st_size, reverse=True)
    if not candidates:
        raise RuntimeError("yt-dlp reported success but produced no file")

    meta = {
        "title": info.get("title") or url,
        "uploader": info.get("uploader"),
        "webpage_url": info.get("webpage_url") or url,
        "source_duration": info.get("duration"),
    }
    return candidates[0], meta


def run(job: Job) -> dict[str, Any]:
    """Resolve the job's source into `{video_path, audio_path, ...probe fields}`."""
    with StageReporter(job, "ingest") as reporter:
        if not media.has_ffmpeg():
            raise media.MediaError(
                "ffmpeg/ffprobe not found on PATH. Install with `brew install ffmpeg`."
            )

        meta: dict[str, Any] = {}
        if job.source == "url":
            reporter.progress(0.02, "Resolving URL…")
            src_path, meta = download(job.source_label, job.id, reporter)
            title = meta.get("title") or src_path.stem
        else:
            src_path = Path(job.media["input_path"])
            if not src_path.exists():
                raise FileNotFoundError(f"uploaded file vanished: {src_path}")
            reporter.progress(0.7, "Reading uploaded file…")
            title = src_path.stem

        reporter.check_cancelled()
        reporter.progress(0.75, "Probing streams…")
        info = media.probe(src_path)
        if not info["has_audio"]:
            raise ValueError("This file has no audio track, so there is nothing to transcribe.")
        if info["duration"] <= 0:
            raise ValueError("Could not determine media duration — the file may be corrupt.")

        reporter.check_cancelled()
        reporter.progress(0.8, "Extracting 16 kHz mono audio…")
        audio_path = media.extract_audio(src_path, config.MEDIA_DIR / job.id / "audio.wav")

        poster = None
        if info["has_video"]:
            reporter.progress(0.94, "Grabbing poster frame…")
            poster = media.thumbnail(
                src_path,
                config.MEDIA_DIR / job.id / "poster.jpg",
                at=min(5.0, info["duration"] / 2),
            )

        result = {
            **info,
            **meta,
            "title": title,
            "video_path": str(src_path),
            "audio_path": str(audio_path),
            "poster": str(poster) if poster else None,
        }
        reporter.log(
            f"Ingested “{title}” — {info['duration'] / 60:.1f} min, "
            f"{info['width']}x{info['height']}"
        )
        return result
