"""Locating the subject so a landscape frame can be cropped to vertical."""
from __future__ import annotations

import pytest

from app.core import framing, media

pytestmark = pytest.mark.skipif(not media.has_ffmpeg(), reason="ffmpeg not on PATH")


def _subject_video(path, subject_x_fraction: float, duration: int = 6):
    """A dark frame with a detailed, moving 'subject' at a known position.

    testsrc2 supplies texture and motion; the rest of the frame is flat, which
    is what a real background looks like to a detail/motion detector.
    """
    w, h = 1280, 720
    sub_w = 200
    x = int(subject_x_fraction * w - sub_w / 2)
    media.run([
        media.binary("ffmpeg"), "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", f"color=c=0x101010:s={w}x{h}:r=10:d={duration}",
        "-f", "lavfi", "-i", f"testsrc2=s={sub_w}x400:r=10:d={duration}",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}",
        "-filter_complex", f"[0:v][1:v]overlay={x}:160:format=auto[v]",
        "-map", "[v]", "-map", "2:a",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", str(path),
    ])
    return path


# --- crop geometry ---

def test_crop_geometry_fills_the_output_aspect():
    box = framing.crop_geometry(1920, 1080, 1080, 1920, centre=0.5)
    assert box is not None
    # 9:16 of a 1080-tall frame is 607.5 -> even 606/608.
    assert 600 <= box["width"] <= 610
    assert box["height"] == 1080
    assert box["y"] == 0
    assert box["width"] % 2 == 0


def test_crop_geometry_follows_the_subject():
    left = framing.crop_geometry(1920, 1080, 1080, 1920, centre=0.2)
    right = framing.crop_geometry(1920, 1080, 1080, 1920, centre=0.8)
    assert left["x"] < right["x"]
    # A subject at 20% should put the crop in the left half.
    assert left["x"] + left["width"] / 2 < 1920 / 2


def test_crop_geometry_stays_inside_the_frame():
    for centre in (-5.0, 0.0, 0.01, 0.99, 1.0, 9.0):
        box = framing.crop_geometry(1920, 1080, 1080, 1920, centre=centre)
        assert box["x"] >= 0
        assert box["x"] + box["width"] <= 1920


def test_crop_geometry_none_when_source_is_not_wider():
    # Already portrait: nothing to crop horizontally.
    assert framing.crop_geometry(1080, 1920, 1080, 1920, 0.5) is None
    # Square source is narrower than 16:9 but taller than 9:16 -> still crops.
    assert framing.crop_geometry(1080, 1080, 1080, 1920, 0.5) is not None


def test_crop_geometry_rejects_nonsense_dimensions():
    assert framing.crop_geometry(0, 1080, 1080, 1920, 0.5) is None
    assert framing.crop_geometry(1920, 0, 1080, 1920, 0.5) is None
    assert framing.crop_geometry(1920, 1080, 0, 1920, 0.5) is None


def test_crop_geometry_centres_when_no_subject_is_known():
    box = framing.crop_geometry(1920, 1080, 1080, 1920, centre=None)
    assert abs((box["x"] + box["width"] / 2) - 960) < 4


# --- window search ---

def test_best_window_finds_a_peak():
    import numpy as np
    scores = np.zeros(300, dtype=np.float32)
    scores[200:240] = 1.0            # a clear subject right of centre
    centre = framing.best_window(scores, window_fraction=0.2)
    assert 0.55 < centre < 0.85, centre


def test_best_window_prefers_centre_when_flat():
    import numpy as np
    centre = framing.best_window(np.ones(300, dtype=np.float32), 0.3)
    assert abs(centre - 0.5) < 0.1


def test_best_window_handles_a_window_wider_than_the_frame():
    import numpy as np
    assert framing.best_window(np.ones(10, dtype=np.float32), 5.0) == 0.5


# --- sampling ---

def test_sample_times_spread_across_all_spans():
    times = framing._sample_times([(0.0, 10.0), (100.0, 110.0)], 8)
    assert len(times) == 8
    assert any(t < 10.0 for t in times), "no samples from the first span"
    assert any(t >= 100.0 for t in times), "no samples from the second span"
    assert all(0.0 <= t <= 110.0 for t in times)


def test_sample_times_degenerate():
    assert framing._sample_times([], 5) == []
    assert framing._sample_times([(0.0, 10.0)], 0) == []


# --- end to end against real pixels ---

@pytest.mark.parametrize("position", [0.25, 0.75])
def test_finds_an_off_centre_subject(tmp_path, position):
    video = _subject_video(tmp_path / f"subj-{position}.mp4", position)
    found = framing.find_subject_x(
        video, [(0.5, 5.5)], window=framing.window_fraction(1280, 720, 1080, 1920), samples=8, work_dir=tmp_path
    )
    assert found is not None, "subject not located"
    # Centre bias pulls the result inwards, so check it leans the right way.
    assert (found < 0.5) == (position < 0.5), \
        f"subject at {position} detected at {found:.2f}"


def test_crop_box_from_a_detected_subject_contains_it(tmp_path):
    video = _subject_video(tmp_path / "subj.mp4", 0.75)
    found = framing.find_subject_x(
        video, [(0.5, 5.5)], window=framing.window_fraction(1280, 720, 1080, 1920), samples=8, work_dir=tmp_path
    )
    box = framing.crop_geometry(1280, 720, 1080, 1920, found)
    subject_px = 0.75 * 1280
    assert box["x"] <= subject_px <= box["x"] + box["width"], \
        f"subject at {subject_px:.0f}px outside crop {box}"


def test_find_subject_returns_none_without_frames(tmp_path):
    missing = tmp_path / "nope.mp4"
    assert framing.find_subject_x(
        missing, [(0.0, 5.0)], window=framing.window_fraction(1280, 720, 1080, 1920), samples=4, work_dir=tmp_path
    ) is None


def test_find_subject_cleans_up_its_frames(tmp_path):
    video = _subject_video(tmp_path / "subj.mp4", 0.5)
    framing.find_subject_x(video, [(0.5, 5.5)], window=framing.window_fraction(1280, 720, 1080, 1920), samples=4,
                           work_dir=tmp_path)
    assert not (tmp_path / "_framing").exists()
