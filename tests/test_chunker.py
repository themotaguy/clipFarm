"""Stage 3 — sliding-window construction and span math."""
from __future__ import annotations

import config
from app.core.chunker import MAX_CHUNK_CHARS, WindowBuilder, _iou, overlap_ratio
from tests.conftest import make_segment


# --- span helpers ---

def test_overlap_ratio_uses_shorter_span():
    # 5s of overlap against a 10s span and a 20s span -> half the *shorter* one.
    assert overlap_ratio((0, 10), (5, 25)) == 0.5


def test_overlap_ratio_disjoint_is_zero():
    assert overlap_ratio((0, 10), (20, 30)) == 0.0


def test_overlap_ratio_zero_length_span():
    assert overlap_ratio((5, 5), (0, 10)) == 0.0


def test_iou_identical_spans_is_one():
    assert _iou((10, 50), (10, 50)) == 1.0


def test_iou_half_overlap():
    # intersection 10, union 30
    assert abs(_iou((0, 20), (10, 30)) - (10 / 30)) < 1e-9


# --- windowing ---

def test_windows_cover_transcript_in_order(transcript):
    builder = WindowBuilder(target=20.0, min_seconds=8.0, max_seconds=40.0)
    windows = []
    for seg in transcript:
        windows.extend(builder.push(seg))
    windows.extend(builder.finish())

    assert windows, "expected at least one window"
    assert [w["start"] for w in windows] == sorted(w["start"] for w in windows)
    for w in windows:
        assert w["end"] > w["start"]
        assert w["text"].strip()
        assert w["duration"] == round(w["end"] - w["start"], 3)


def test_window_ids_are_unique(transcript):
    builder = WindowBuilder(target=20.0, min_seconds=8.0, max_seconds=40.0)
    windows = []
    for seg in transcript:
        windows.extend(builder.push(seg))
    windows.extend(builder.finish())
    ids = [w["id"] for w in windows]
    assert len(ids) == len(set(ids)), f"duplicate window ids: {ids}"


def test_window_ids_unique_when_text_is_split(transcript):
    """A window whose text exceeds MAX_CHUNK_CHARS is split into several pieces.

    Each piece is a separate Chroma document, so each needs its own id.
    """
    # 60 words per ~1.2s of speech pushes each window well past MAX_CHUNK_CHARS.
    dense = []
    t = 0.0
    for i in range(14):
        text = " ".join(f"word{i:02d}x{j:02d}" for j in range(60))
        seg = make_segment(i, t, text, word_dur=0.02)
        dense.append(seg)
        t = seg["end"] + 0.05

    builder = WindowBuilder(target=10.0, min_seconds=1.0, max_seconds=30.0)
    windows = []
    for seg in dense:
        windows.extend(builder.push(seg))
    windows.extend(builder.finish())

    # Guard the guard: this fixture must actually trigger the splitter, otherwise
    # the test would pass vacuously.
    assert len(windows) > 1
    assert any(len(w["text"]) >= MAX_CHUNK_CHARS - 200 for w in windows), \
        "fixture no longer exercises the oversized-window split path"
    ids = [w["id"] for w in windows]
    assert len(ids) == len(set(ids)), f"duplicate window ids after splitting: {ids}"
    indices = [w["index"] for w in windows]
    assert len(indices) == len(set(indices)), f"duplicate window indices: {indices}"


def test_windows_respect_max_seconds(transcript):
    builder = WindowBuilder(target=15.0, min_seconds=5.0, max_seconds=18.0)
    windows = []
    for seg in transcript:
        windows.extend(builder.push(seg))
    windows.extend(builder.finish())
    for w in windows:
        # One pathological segment can exceed the cap; our fixture has none.
        assert w["duration"] <= 18.0 + 5.0


def test_consecutive_windows_overlap(transcript):
    builder = WindowBuilder(target=20.0, min_seconds=8.0, max_seconds=40.0, overlap=0.5)
    windows = []
    for seg in transcript:
        windows.extend(builder.push(seg))
    windows.extend(builder.finish())
    if len(windows) >= 2:
        assert windows[1]["start"] < windows[0]["end"], "windows should overlap in time"


def test_empty_input_produces_nothing():
    builder = WindowBuilder()
    assert builder.finish() == []


def test_single_short_segment_still_emits():
    builder = WindowBuilder(target=45.0, min_seconds=18.0, max_seconds=75.0)
    seg = make_segment(0, 0.0, "A very short remark that stands alone.")
    assert builder.push(seg) == []          # below target, nothing completed yet
    out = builder.finish()
    assert len(out) == 1                    # first window is emitted even if short
    assert out[0]["start"] == seg["start"]


def test_segments_without_word_timings_fall_back_to_segment_span():
    builder = WindowBuilder(target=10.0, min_seconds=2.0, max_seconds=30.0)
    segs = [
        {"id": 0, "start": 0.0, "end": 6.0, "text": "First chunk of speech.", "words": []},
        {"id": 1, "start": 6.0, "end": 12.0, "text": "Second chunk of speech.", "words": []},
    ]
    windows = []
    for s in segs:
        windows.extend(builder.push(s))
    windows.extend(builder.finish())
    assert windows
    assert windows[0]["start"] == 0.0


def test_builder_defaults_come_from_config():
    builder = WindowBuilder()
    assert builder.target == config.CLIP_TARGET_SECONDS
    assert builder.min_seconds == config.CLIP_MIN_SECONDS
    assert builder.max_seconds == config.CLIP_MAX_SECONDS
