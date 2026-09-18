"""Stage 6 — real ffmpeg renders, kept deliberately tiny."""
from __future__ import annotations

import pytest

import config
from app.core import media, pngcaptions, render, timeline
from tests.conftest import make_segment

pytestmark = pytest.mark.skipif(not media.has_ffmpeg(), reason="ffmpeg not on PATH")


@pytest.fixture(scope="module")
def source_video(tmp_path_factory):
    path = tmp_path_factory.mktemp("render") / "src.mp4"
    media.run([
        media.binary("ffmpeg"), "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=15:duration=30",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=30",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", str(path),
    ])
    return path


@pytest.fixture
def segments():
    """Speech across the whole source, so any span has captions."""
    segs, t = [], 1.0
    for i in range(12):
        seg = make_segment(i, t, f"Line {i} of speech for the render test here.")
        segs.append(seg)
        t = seg["end"] + 0.4          # a gap wide enough to count as a pause
    return segs


# --- single span ---

def test_renders_vertical_clip_with_sidecars(source_video, segments, tmp_path):
    out = tmp_path / "clip-01.mp4"
    info = render.render_clip(
        source_video, out, [(1.0, 5.0)], segments,
        has_video=True, width=270, height=480,
    )

    assert out.exists() and info["size_bytes"] > 0
    assert (info["width"], info["height"]) == (270, 480)
    assert 3.5 < info["rendered_duration"] < 4.6
    assert info["stitched"] is False
    assert info["spans"] == [{"start": 1.0, "end": 5.0}]
    # Sidecars sit next to the video for re-upload elsewhere.
    assert (tmp_path / "clip-01.srt").exists()
    assert (tmp_path / "clip-01.ass").exists()
    assert info["thumbnail"] and info["thumbnail"].endswith(".jpg")


def test_accepts_a_bare_start_end_pair(source_video, segments, tmp_path):
    info = render.render_clip(
        source_video, tmp_path / "clip.mp4", (2.0, 5.0), segments,
        has_video=True, width=270, height=480,
    )
    assert info["spans"] == [{"start": 2.0, "end": 5.0}]


def test_srt_sidecar_is_rebased_to_clip_start(source_video, segments, tmp_path):
    render.render_clip(
        source_video, tmp_path / "clip.mp4", [(4.0, 8.0)], segments,
        has_video=True, width=270, height=480,
    )
    srt = (tmp_path / "clip.srt").read_text()
    assert srt.strip(), "expected caption lines in range"
    assert "00:00:0" in srt.splitlines()[1]


# --- stitching ---

def test_stitches_two_spans_into_one_clip(source_video, segments, tmp_path):
    spans = [(1.0, 4.0), (20.0, 24.0)]
    info = render.render_clip(
        source_video, tmp_path / "stitch.mp4", spans, segments,
        has_video=True, width=270, height=480,
    )
    assert info["stitched"] is True
    assert len(info["spans"]) == 2
    # Duration is the sum of the spans, not the 23s envelope.
    assert abs(info["rendered_duration"] - timeline.total_duration(spans)) < 0.5


def test_adjacent_spans_are_fused_into_one_cut(source_video, segments, tmp_path):
    info = render.render_clip(
        source_video, tmp_path / "fused.mp4", [(1.0, 4.0), (4.0, 7.0)], segments,
        has_video=True, width=270, height=480,
    )
    assert info["stitched"] is False
    assert info["spans"] == [{"start": 1.0, "end": 7.0}]


def test_stitched_clip_captions_cover_both_spans(source_video, segments, tmp_path):
    render.render_clip(
        source_video, tmp_path / "stitch.mp4", [(1.0, 5.0), (20.0, 24.0)], segments,
        has_video=True, width=270, height=480,
    )
    srt = (tmp_path / "stitch.srt").read_text()
    assert srt.count("-->") >= 2
    # Nothing may be timed beyond the 8s stitched length.
    assert "00:00:2" not in srt.split("-->")[-1]


def test_render_rejects_empty_spans(source_video, segments, tmp_path):
    with pytest.raises(media.MediaError):
        render.render_clip(
            source_video, tmp_path / "none.mp4", [], segments,
            has_video=True, width=270, height=480,
        )


# --- caption backends ---

def test_backend_auto_prefers_libass_when_available(monkeypatch):
    monkeypatch.setattr(config, "CAPTION_RENDERER", "auto")
    monkeypatch.setattr(media, "has_filter", lambda name: True)
    assert render.caption_backend(True) == "libass"


def test_backend_auto_falls_back_to_pillow(monkeypatch):
    monkeypatch.setattr(config, "CAPTION_RENDERER", "auto")
    monkeypatch.setattr(media, "has_filter", lambda name: False)
    monkeypatch.setattr(pngcaptions, "available", lambda: True)
    assert render.caption_backend(True) == "pillow"


def test_backend_none_when_nothing_can_draw(monkeypatch):
    monkeypatch.setattr(config, "CAPTION_RENDERER", "auto")
    monkeypatch.setattr(media, "has_filter", lambda name: False)
    monkeypatch.setattr(pngcaptions, "available", lambda: False)
    assert render.caption_backend(True) == "none"


def test_backend_respects_burn_captions_off():
    assert render.caption_backend(False) == "none"


def test_backend_libass_forced_but_missing(monkeypatch):
    monkeypatch.setattr(config, "CAPTION_RENDERER", "libass")
    monkeypatch.setattr(media, "has_filter", lambda name: False)
    assert render.caption_backend(True) == "none"


@pytest.mark.skipif(not media.has_filter("subtitles"),
                    reason="this ffmpeg has no libass")
def test_libass_backend_burns_captions(source_video, segments, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CAPTION_RENDERER", "libass")
    info = render.render_clip(
        source_video, tmp_path / "libass.mp4", [(1.0, 5.0)], segments,
        has_video=True, width=270, height=480,
    )
    assert info["caption_backend"] == "libass"
    assert info["captions_burned"] is True
    assert info["captions_skipped_reason"] is None


@pytest.mark.skipif(not pngcaptions.available(), reason="Pillow not installed")
def test_pillow_backend_burns_captions(source_video, segments, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CAPTION_RENDERER", "pillow")
    info = render.render_clip(
        source_video, tmp_path / "pillow.mp4", [(1.0, 5.0)], segments,
        has_video=True, width=270, height=480,
    )
    assert info["caption_backend"] == "pillow"
    assert info["captions_burned"] is True
    # The intermediate PNGs must not be left behind.
    assert not (tmp_path / "pillow-captions").exists()


def test_captions_off_reports_no_reason(source_video, segments, tmp_path):
    info = render.render_clip(
        source_video, tmp_path / "plain.mp4", [(1.0, 4.0)], segments,
        has_video=True, width=270, height=480, burn_captions=False,
    )
    assert info["captions_burned"] is False
    assert info["caption_backend"] == "none"
    assert info["captions_skipped_reason"] is None


# --- audio-only ---

def test_audio_only_source_renders_a_waveform(source_video, segments, tmp_path):
    audio = media.extract_audio(source_video, tmp_path / "audio.wav")
    info = render.render_clip(
        audio, tmp_path / "audio-clip.mp4", [(1.0, 4.0)], segments,
        has_video=False, width=270, height=480,
    )
    assert (info["width"], info["height"]) == (270, 480)
    assert info["size_bytes"] > 0


def test_audio_only_source_can_stitch(source_video, segments, tmp_path):
    audio = media.extract_audio(source_video, tmp_path / "audio.wav")
    spans = [(1.0, 4.0), (20.0, 23.0)]
    info = render.render_clip(
        audio, tmp_path / "audio-stitch.mp4", spans, segments,
        has_video=False, width=270, height=480,
    )
    assert info["stitched"] is True
    assert abs(info["rendered_duration"] - timeline.total_duration(spans)) < 0.6


def test_render_raises_on_missing_source(segments, tmp_path):
    with pytest.raises(media.MediaError):
        render.render_clip(
            tmp_path / "nope.mp4", tmp_path / "out.mp4", [(0.0, 2.0)], segments,
            has_video=True, width=270, height=480,
        )


# --- filling the vertical frame ---

@pytest.fixture(scope="module")
def landscape_video(tmp_path_factory):
    """16:9 source with a detailed subject left of centre."""
    path = tmp_path_factory.mktemp("fill") / "wide.mp4"
    media.run([
        media.binary("ffmpeg"), "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=c=0x101010:s=1280x720:r=10:d=12",
        "-f", "lavfi", "-i", "testsrc2=s=200x400:r=10:d=12",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=12",
        "-filter_complex", "[0:v][1:v]overlay=220:160:format=auto[v]",
        "-map", "[v]", "-map", "2:a",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", str(path),
    ])
    return path


def test_crop_fill_locates_the_subject(landscape_video, segments, tmp_path):
    info = render.render_clip(
        landscape_video, tmp_path / "crop.mp4", [(1.0, 5.0)], segments,
        has_video=True, width=270, height=480, fill="crop",
    )
    assert info["fill"] == "crop"
    assert info["crop"] is not None
    assert info["subject_x"] is not None
    # Subject sits at ~25% of the width, so the crop must lean left.
    assert info["subject_x"] < 0.5, info["subject_x"]
    assert (info["width"], info["height"]) == (270, 480)


def test_crop_fill_box_stays_inside_the_source(landscape_video, segments, tmp_path):
    info = render.render_clip(
        landscape_video, tmp_path / "crop.mp4", [(1.0, 4.0)], segments,
        has_video=True, width=270, height=480, fill="crop",
    )
    box = info["crop"]
    assert box["x"] >= 0
    assert box["x"] + box["width"] <= 1280
    assert box["height"] == 720


def test_blur_fill_does_not_crop(landscape_video, segments, tmp_path):
    info = render.render_clip(
        landscape_video, tmp_path / "blur.mp4", [(1.0, 4.0)], segments,
        has_video=True, width=270, height=480, fill="blur",
    )
    assert info["fill"] == "blur"
    assert info["crop"] is None
    assert info["subject_x"] is None


def test_fill_defaults_to_the_configured_mode(landscape_video, segments,
                                               tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RENDER_FILL", "blur")
    info = render.render_clip(
        landscape_video, tmp_path / "default.mp4", [(1.0, 4.0)], segments,
        has_video=True, width=270, height=480,
    )
    assert info["fill"] == "blur"


def test_unknown_fill_mode_falls_back_to_blur(landscape_video, segments, tmp_path):
    info = render.render_clip(
        landscape_video, tmp_path / "weird.mp4", [(1.0, 4.0)], segments,
        has_video=True, width=270, height=480, fill="nonsense",
    )
    assert info["fill"] == "blur"


def test_crop_fill_works_on_a_stitched_clip(landscape_video, segments, tmp_path):
    spans = [(1.0, 4.0), (8.0, 11.0)]
    info = render.render_clip(
        landscape_video, tmp_path / "crop-stitch.mp4", spans, segments,
        has_video=True, width=270, height=480, fill="crop",
    )
    assert info["stitched"] is True
    assert info["fill"] == "crop"
    assert abs(info["rendered_duration"] - timeline.total_duration(spans)) < 0.6


def test_crop_fill_on_a_portrait_source_covers_instead(segments, tmp_path):
    """Nothing to crop horizontally, so it must cover rather than fail."""
    portrait = tmp_path / "tall.mp4"
    media.run([
        media.binary("ffmpeg"), "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc2=s=480x854:r=10:d=6",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=6",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", str(portrait),
    ])
    info = render.render_clip(
        portrait, tmp_path / "tall-out.mp4", [(1.0, 4.0)], segments,
        has_video=True, width=270, height=480, fill="crop",
    )
    assert info["crop"] is None          # no horizontal crop needed
    assert (info["width"], info["height"]) == (270, 480)


def test_audio_only_ignores_fill_mode(source_video, segments, tmp_path):
    audio = media.extract_audio(source_video, tmp_path / "a.wav")
    info = render.render_clip(
        audio, tmp_path / "a.mp4", [(1.0, 4.0)], segments,
        has_video=False, width=270, height=480, fill="crop",
    )
    assert info["crop"] is None
    assert info["size_bytes"] > 0
