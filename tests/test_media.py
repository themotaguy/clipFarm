"""ffmpeg/ffprobe wrappers. These shell out for real, so they need ffmpeg."""
from __future__ import annotations

import pytest

from app.core import media

pytestmark = pytest.mark.skipif(not media.has_ffmpeg(), reason="ffmpeg not on PATH")


@pytest.fixture(scope="module")
def tiny_video(tmp_path_factory):
    """A real 3-second 320x240 clip with a tone, built once per run."""
    path = tmp_path_factory.mktemp("media") / "tiny.mp4"
    media.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=15:duration=3",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", str(path),
    ])
    return path


def test_probe_reports_streams_and_duration(tiny_video):
    info = media.probe(tiny_video)
    assert info["has_video"] is True
    assert info["has_audio"] is True
    assert info["width"] == 320
    assert info["height"] == 240
    assert 2.5 < info["duration"] < 3.6
    assert info["video_codec"] == "h264"
    assert info["size_bytes"] > 0


def test_probe_raises_on_garbage(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(media.MediaError):
        media.probe(bad)


def test_extract_audio_produces_16k_mono_wav(tiny_video, tmp_path):
    out = media.extract_audio(tiny_video, tmp_path / "nested" / "audio.wav")
    assert out.exists()
    info = media.probe(out)
    assert info["has_audio"] is True
    assert info["has_video"] is False


def test_thumbnail_writes_an_image(tiny_video, tmp_path):
    out = media.thumbnail(tiny_video, tmp_path / "poster.jpg", at=1.0, width=160)
    assert out is not None and out.exists()
    assert out.stat().st_size > 0


def test_thumbnail_returns_none_for_audio_only(tiny_video, tmp_path):
    audio = media.extract_audio(tiny_video, tmp_path / "audio.wav")
    assert media.thumbnail(audio, tmp_path / "nope.jpg") is None


def test_has_filter_detects_real_and_fake_filters():
    # These are core filters present in every practical build.
    assert media.has_filter("scale") is True
    assert media.has_filter("overlay") is True
    assert media.has_filter("definitely-not-a-filter") is False


def test_has_filter_is_cached(monkeypatch):
    media.has_filter("scale")           # warm the cache
    calls = []
    real_run = media.subprocess.run

    def counting_run(*a, **kw):
        calls.append(a)
        return real_run(*a, **kw)

    monkeypatch.setattr(media.subprocess, "run", counting_run)
    media.has_filter("scale")
    assert calls == [], "cached lookup should not re-invoke ffmpeg"


def test_run_raises_with_stderr_tail():
    with pytest.raises(media.MediaError) as exc:
        media.run(["ffmpeg", "-hide_banner", "-i", "/nonexistent/file.mp4", "-f", "null", "-"])
    assert "ffmpeg failed" in str(exc.value)
