"""Source-time <-> clip-time mapping for stitched clips."""
from __future__ import annotations

from app.core import timeline
from tests.conftest import make_segment


def test_normalize_accepts_pairs_dicts_and_sorts():
    out = timeline.normalize([
        (30.0, 40.0),
        {"start": 10.0, "end": 20.0},
        [50.0, 55.0],
    ])
    assert out == [(10.0, 20.0), (30.0, 40.0), (50.0, 55.0)]


def test_normalize_drops_empty_and_inverted_spans():
    assert timeline.normalize([(5.0, 5.0), (9.0, 3.0), (1.0, 2.0)]) == [(1.0, 2.0)]
    assert timeline.normalize([]) == []
    assert timeline.normalize(None) == []


def test_total_duration_sums_spans_not_their_envelope():
    spans = [(10.0, 20.0), (100.0, 108.0)]
    assert timeline.total_duration(spans) == 18.0


def test_offsets_are_cumulative():
    assert timeline.offsets([(10.0, 20.0), (100.0, 105.0), (200.0, 203.0)]) == [
        0.0, 10.0, 15.0,
    ]


def test_to_clip_time_maps_each_span():
    spans = [(10.0, 20.0), (100.0, 110.0)]
    assert timeline.to_clip_time(spans, 10.0) == 0.0
    assert timeline.to_clip_time(spans, 15.0) == 5.0
    assert timeline.to_clip_time(spans, 100.0) == 10.0
    assert timeline.to_clip_time(spans, 105.0) == 15.0
    # A time in the excised gap belongs to no span.
    assert timeline.to_clip_time(spans, 50.0) is None


def test_clamp_to_clip_attaches_gaps_to_the_nearest_edge():
    spans = [(10.0, 20.0), (100.0, 110.0)]
    assert timeline.clamp_to_clip(spans, 21.0) == 10.0    # just after span 1
    assert timeline.clamp_to_clip(spans, 99.0) == 10.0    # just before span 2
    assert timeline.clamp_to_clip(spans, 0.0) == 0.0
    assert timeline.clamp_to_clip([], 5.0) == 0.0


# --- content rebasing ---

def make_transcript():
    segs, t = [], 0.0
    for i in range(30):
        seg = make_segment(i, t, f"Line number {i} of the transcript here.")
        segs.append(seg)
        t = seg["end"] + 0.1
    return segs


def test_words_in_spans_rebases_onto_the_stitched_timeline():
    segs = make_transcript()
    # Two disjoint windows; the second must land immediately after the first.
    spans = [(0.0, 8.0), (40.0, 46.0)]
    words = timeline.words_in_spans(segs, spans)

    assert words
    assert min(w["start"] for w in words) >= 0.0
    assert max(w["end"] for w in words) <= timeline.total_duration(spans) + 0.01
    assert [w["start"] for w in words] == sorted(w["start"] for w in words)
    # Nothing may land in the excised region: span 1 is 8s long, so the second
    # span's content starts at clip time 8.0, not at 40.0.
    assert any(abs(w["start"] - 8.0) < 2.0 for w in words)


def test_words_in_spans_clips_words_straddling_a_cut():
    segs = [make_segment(0, 0.0, "one two three four five six seven eight")]
    spans = [(0.0, 1.0)]
    words = timeline.words_in_spans(segs, spans)
    assert words
    assert all(w["end"] <= 1.0 + 1e-9 for w in words)


def test_words_in_spans_empty_for_disjoint_span():
    segs = make_transcript()
    assert timeline.words_in_spans(segs, [(10_000.0, 10_010.0)]) == []


def test_segments_in_spans_rebases_and_orders():
    segs = make_transcript()
    spans = [(0.0, 8.0), (40.0, 46.0)]
    out = timeline.segments_in_spans(segs, spans)
    assert out
    assert all(s["end"] > s["start"] for s in out)
    assert max(s["end"] for s in out) <= timeline.total_duration(spans) + 0.01
    assert [s["start"] for s in out] == sorted(s["start"] for s in out)


def test_single_span_behaves_like_a_plain_offset():
    segs = make_transcript()
    words = timeline.words_in_spans(segs, [(20.0, 30.0)])
    assert words
    assert abs(min(w["start"] for w in words)) < 1.0
    assert max(w["end"] for w in words) <= 10.01


# --- merging ---

def test_merge_touching_fuses_adjacent_spans():
    assert timeline.merge_touching([(0.0, 10.0), (10.0, 20.0)]) == [(0.0, 20.0)]
    assert timeline.merge_touching([(0.0, 10.0), (10.02, 20.0)]) == [(0.0, 20.0)]


def test_merge_touching_keeps_distant_spans_apart():
    assert timeline.merge_touching([(0.0, 10.0), (50.0, 60.0)]) == [
        (0.0, 10.0), (50.0, 60.0),
    ]


def test_merge_touching_absorbs_a_contained_span():
    assert timeline.merge_touching([(0.0, 30.0), (5.0, 10.0)]) == [(0.0, 30.0)]
