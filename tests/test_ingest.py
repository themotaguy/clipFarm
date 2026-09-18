"""Stage 1 — resolving a job source into local media."""
from __future__ import annotations

import pytest

import config
from app.core import ingest, media
from app.jobs import Job

pytestmark = pytest.mark.skipif(not media.has_ffmpeg(), reason="ffmpeg not on PATH")


def test_is_url_accepts_only_http_schemes():
    assert ingest.is_url("https://example.com/v") is True
    assert ingest.is_url("  http://example.com/v  ") is True
    assert ingest.is_url("file:///etc/passwd") is False
    assert ingest.is_url("example.com/v") is False
    assert ingest.is_url("") is False


@pytest.fixture
def upload_job(tmp_path, monkeypatch):
    """A job whose upload lives at `source.mp4`, as the API saves it."""
    monkeypatch.setattr(config, "MEDIA_DIR", tmp_path / "media")
    job = Job(id="job1", source="upload", source_label="My Big Talk.mp4")
    dest = tmp_path / "uploads" / "job1"
    dest.mkdir(parents=True)
    src = dest / "source.mp4"
    media.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=15:duration=2",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", str(src),
    ])
    job.media["input_path"] = str(src)
    return job


def test_upload_title_uses_the_original_filename(upload_job):
    """The on-disk name is always `source.mp4`; the label must survive ingest."""
    info = ingest.run(upload_job)
    assert info["title"] == "My Big Talk"


def test_ingest_returns_paths_and_probe_fields(upload_job):
    info = ingest.run(upload_job)
    assert info["has_audio"] is True
    assert info["has_video"] is True
    assert info["duration"] > 1.0
    assert info["video_path"] == upload_job.media["input_path"]
    assert info["audio_path"].endswith("audio.wav")
    assert info["poster"] and info["poster"].endswith("poster.jpg")


def test_ingest_rejects_missing_upload(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MEDIA_DIR", tmp_path / "media")
    job = Job(id="job2", source="upload", source_label="gone.mp4")
    job.media["input_path"] = str(tmp_path / "does-not-exist.mp4")
    with pytest.raises(FileNotFoundError):
        ingest.run(job)


def test_ingest_rejects_media_without_audio(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MEDIA_DIR", tmp_path / "media")
    silent = tmp_path / "source.mp4"
    media.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=15:duration=2",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        str(silent),
    ])
    job = Job(id="job3", source="upload", source_label="silent.mp4")
    job.media["input_path"] = str(silent)
    with pytest.raises(ValueError, match="no audio track"):
        ingest.run(job)
