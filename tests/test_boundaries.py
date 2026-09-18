"""Cut-point selection — pauses, not word edges."""
from __future__ import annotations

from app.core import boundaries


def words_from(spec: list[tuple[str, float, float]]) -> list[dict]:
    return [{"word": w, "start": s, "end": e} for w, s, e in spec]


# A short passage with three distinct pauses:
#   after "here."  (2.00 -> 2.60, sentence end, 0.60s)
#   after "next"   (4.00 -> 4.30, mid-sentence,  0.30s)
#   after "done."  (5.50 -> 6.40, sentence end, 0.90s)
PASSAGE = words_from([
    ("This", 0.0, 0.4), ("is", 0.4, 0.7), ("here.", 0.7, 2.0),
    ("Then", 2.6, 3.0), ("comes", 3.0, 3.6), ("next", 3.6, 4.0),
    ("and", 4.3, 4.6), ("we", 4.6, 4.9), ("are", 4.9, 5.2), ("done.", 5.2, 5.5),
    ("After", 6.4, 6.9), ("that", 6.9, 7.4),
])


def test_flatten_words_orders_and_skips_untimed():
    segs = [
        {"words": [{"word": "b", "start": 1.0, "end": 1.5}]},
        {"words": [{"word": "a", "start": 0.0, "end": 0.5},
                   {"word": "bad", "start": None, "end": 2.0}]},
        {"words": []},
        {},
    ]
    out = boundaries.flatten_words(segs)
    assert [w["word"] for w in out] == ["a", "b"]


def test_sentence_and_clause_detection():
    assert boundaries.ends_sentence({"word": "done."}) is True
    assert boundaries.ends_sentence({"word": "really?"}) is True
    assert boundaries.ends_sentence({"word": " wow! "}) is True
    assert boundaries.ends_sentence({"word": "next"}) is False
    assert boundaries.ends_clause({"word": "however,"}) is True
    assert boundaries.ends_clause({"word": "next"}) is False


def test_find_pauses_only_returns_real_gaps():
    pauses = boundaries.find_pauses(PASSAGE, min_pause=0.2)
    spans = [(round(p.start, 2), round(p.end, 2)) for p in pauses]
    assert (2.0, 2.6) in spans
    assert (4.0, 4.3) in spans
    assert (5.5, 6.4) in spans
    # Gaps of exactly 0.0 between consecutive words are not pauses.
    assert (0.4, 0.4) not in spans


def test_find_pauses_marks_sentence_boundaries():
    by_start = {round(p.start, 2): p for p in boundaries.find_pauses(PASSAGE, 0.2)}
    assert by_start[2.0].after_sentence is True
    assert by_start[4.0].after_sentence is False
    assert by_start[4.0].after_clause is False


def test_find_pauses_does_not_invent_silence_at_the_edges():
    """`words` may be a fragment, so a pause from t=0 would be fabricated."""
    words = words_from([("only", 3.0, 3.5)])
    assert boundaries.find_pauses(words, 0.2) == []


def test_find_pauses_empty_input():
    assert boundaries.find_pauses([], 0.2) == []


# --- end points ---

def test_snap_end_moves_into_the_pause_not_the_word_edge():
    """With no sentence break in reach, cut inside the nearest mid-sentence pause."""
    # max_shift keeps the 5.5 sentence break out of range, leaving the 4.0->4.3 gap.
    out = boundaries.snap_end(4.0, PASSAGE, max_shift=0.5, min_pause=0.2,
                              tail=0.6, ceiling=100.0)
    assert 4.0 < out <= 4.3, out


def test_snap_end_prefers_a_sentence_pause_over_a_closer_mid_sentence_one():
    # 4.0 is a mid-sentence pause; 5.5 ends a sentence and is within reach.
    out = boundaries.snap_end(4.2, PASSAGE, max_shift=2.5, min_pause=0.2,
                              tail=0.6, ceiling=100.0)
    assert out > 5.5, f"expected the sentence break at 5.5, got {out}"


def test_snap_end_holds_the_line_on_continuous_speech():
    """No silence anywhere to borrow, so the cut cannot be padded at all."""
    lockstep = words_from([(f"w{i}", i * 0.3, (i + 1) * 0.3) for i in range(40)])
    out = boundaries.snap_end(6.0, lockstep, max_shift=0.5, min_pause=0.25,
                              tail=0.6, ceiling=100.0)
    assert abs(out - 6.0) < 1e-6


def test_snap_end_never_exceeds_the_ceiling():
    out = boundaries.snap_end(9.9, PASSAGE, max_shift=2.5, min_pause=0.2,
                              tail=0.6, ceiling=10.0)
    assert out <= 10.0


def test_snap_end_ignores_pauses_beyond_max_shift():
    # Nearest pause starts at 4.0, well outside a 0.05s shift budget from 3.2.
    out = boundaries.snap_end(3.2, PASSAGE, max_shift=0.05, min_pause=0.2,
                              tail=0.6, ceiling=100.0)
    # Padded towards the next word ("next" at 3.6) but not into it.
    assert abs(out - 3.6) < 1e-6


def test_snap_end_prefers_extending_over_truncating():
    """Two equally good pauses, one ahead and one behind -> take the one ahead."""
    words = words_from([
        ("a", 0.0, 1.0), ("b.", 1.0, 2.0),      # pause 2.0 -> 2.8 (sentence)
        ("c", 2.8, 3.8), ("d.", 3.8, 4.8),      # pause 4.8 -> 5.6 (sentence)
        ("e", 5.6, 6.0),
    ])
    out = boundaries.snap_end(3.4, words, max_shift=2.5, min_pause=0.2,
                              tail=0.6, ceiling=100.0)
    assert out > 4.8, f"should reach forward to 4.8, not back to 2.0 (got {out})"


# --- start points ---

def test_snap_start_begins_just_inside_the_preceding_silence():
    # Target is the first word after the 2.0->2.6 pause.
    out = boundaries.snap_start(2.6, PASSAGE, max_shift=2.5, min_pause=0.2,
                                lead_in=0.25)
    assert 2.0 < out < 2.6, out
    assert abs(out - (2.6 - 0.25)) < 1e-6


def test_snap_start_never_goes_below_the_floor():
    out = boundaries.snap_start(0.1, PASSAGE, max_shift=2.5, min_pause=0.2,
                                lead_in=0.5, floor=0.0)
    assert out >= 0.0


def test_snap_start_takes_no_lead_in_from_continuous_speech():
    lockstep = words_from([(f"w{i}", i * 0.3, (i + 1) * 0.3) for i in range(40)])
    out = boundaries.snap_start(6.0, lockstep, max_shift=0.5, min_pause=0.25,
                                lead_in=0.25)
    assert abs(out - 6.0) < 1e-6


# --- whole spans ---

def test_snap_span_produces_breathing_room_on_both_sides():
    start, end = boundaries.snap_span(
        2.6, 4.0, PASSAGE, max_shift=2.5, min_pause=0.2,
        lead_in=0.25, tail=0.6, media_duration=8.0,
    )
    assert start < 2.6, "clip should start before the first word"
    assert end > 4.0, "clip should not end on the final word's edge"
    assert end > start


def test_snap_span_respects_max_seconds():
    start, end = boundaries.snap_span(
        0.5, 7.0, PASSAGE, max_shift=2.5, min_pause=0.2, lead_in=0.25,
        tail=0.6, media_duration=8.0, max_seconds=3.0,
    )
    assert end - start <= 3.0 + 1e-6


def test_snap_span_respects_min_seconds():
    start, end = boundaries.snap_span(
        2.6, 3.0, PASSAGE, max_shift=0.05, min_pause=0.2, lead_in=0.1,
        tail=0.1, media_duration=30.0, min_seconds=5.0,
    )
    assert end - start >= 5.0 - 1e-6


def test_snap_span_keeps_order_when_snapping_collapses_it():
    start, end = boundaries.snap_span(
        5.0, 5.05, [], max_shift=2.5, min_pause=0.2,
        lead_in=0.25, tail=0.6, media_duration=30.0,
    )
    assert end > start


def test_snap_span_with_no_words_still_pads():
    start, end = boundaries.snap_span(
        10.0, 30.0, [], max_shift=2.5, min_pause=0.2,
        lead_in=0.25, tail=0.6, media_duration=60.0,
    )
    assert start < 10.0 and end > 30.0


# --- padding must never intrude into a neighbouring word ---

def test_snap_start_padding_never_enters_the_previous_word():
    """Back-to-back speech: a lead-in would land mid-word, so don't take one."""
    words = words_from([
        ("is", 79.36, 79.52), ("wildly", 79.52, 79.84),
        ("underutilized", 79.84, 80.74), ("by", 80.74, 80.94),
        ("a", 80.94, 81.12), ("single", 81.12, 81.38), ("user.", 81.38, 81.74),
    ])
    out = boundaries.snap_start(80.74, words, max_shift=2.5, min_pause=0.18,
                                lead_in=0.25)
    assert out >= 80.74, f"started inside 'underutilized' (got {out})"


def test_snap_end_padding_never_enters_the_next_word():
    words = words_from([
        ("one", 10.0, 10.5), ("two", 10.5, 11.0), ("three", 11.0, 11.5),
    ])
    out = boundaries.snap_end(11.0, words, max_shift=2.5, min_pause=0.18,
                              tail=0.6, ceiling=100.0)
    assert out <= 11.0, f"cut into 'three' (got {out})"


def test_snap_end_takes_only_the_available_gap():
    # A 0.1s gap is below min_pause, so the tail is limited to that gap.
    words = words_from([("one", 10.0, 10.5), ("two", 10.6, 11.0)])
    out = boundaries.snap_end(10.5, words, max_shift=2.5, min_pause=0.18,
                              tail=0.6, ceiling=100.0)
    assert 10.5 <= out <= 10.6


def test_snap_span_on_continuous_speech_does_not_clip_words():
    words = words_from([(f"w{i}", 10.0 + i * 0.3, 10.0 + (i + 1) * 0.3)
                        for i in range(60)])
    start, end = boundaries.snap_span(
        13.0, 25.0, words, max_shift=1.0, min_pause=0.18,
        lead_in=0.25, tail=0.6, media_duration=60.0,
    )
    straddled = [w for w in words
                 if w["start"] < start < w["end"] or w["start"] < end < w["end"]]
    assert not straddled, f"boundaries land inside words: {straddled}"


def test_snap_end_does_not_truncate_an_already_clean_ending():
    """A target that already ends a sentence must not be pulled backwards.

    Regression: with no pause ahead of the final word, the best of the earlier
    pauses used to win and cut the last sentence in half.
    """
    words = words_from([
        ("Thanks", 0.0, 0.5), ("for", 0.5, 0.8), ("listening.", 0.8, 1.6),
        ("and", 2.0, 2.3), ("I", 2.3, 2.5), ("will", 2.5, 2.9),
        ("see", 2.9, 3.2), ("you", 3.2, 3.5), ("time.", 3.5, 4.2),
    ])
    out = boundaries.snap_end(4.2, words, max_shift=2.5, min_pause=0.18,
                              tail=0.6, ceiling=10.0)
    assert out >= 4.2, f"truncated a clean sentence ending (got {out})"


def test_snap_start_does_not_skip_forward_past_a_clean_opening():
    words = words_from([
        ("Here.", 0.0, 0.6),
        ("The", 1.4, 1.7), ("model", 1.7, 2.1), ("is", 2.1, 2.3),
        ("easy.", 2.3, 2.9), ("But", 3.4, 3.7), ("serving", 3.7, 4.2),
    ])
    # 1.4 already follows a sentence break; don't jump to the one at 2.9.
    out = boundaries.snap_start(1.4, words, max_shift=2.5, min_pause=0.18,
                                lead_in=0.25)
    assert out < 1.6, f"skipped past the intended opening (got {out})"


# --- sentence breaks with no measurable gap ---

def test_sentence_end_counts_even_without_a_pause():
    """Speakers run through full stops, so Whisper leaves almost no gap there.

    Regression from real footage: the cut had to reach '...flagships.' across a
    0.08s gap, which a pause-length test alone would never offer.
    """
    words = words_from([
        ("any", 0.0, 0.3), ("testing,", 0.3, 0.6),
        ("probably", 0.72, 1.1), ("the", 1.1, 1.3), ("flagships.", 1.3, 1.9),
        ("And", 1.98, 2.1), ("then", 2.1, 2.3),
    ])
    pauses = boundaries.find_pauses(words, min_pause=0.18)
    assert any(abs(p.start - 1.9) < 1e-6 and p.after_sentence for p in pauses), \
        "the sentence boundary after 'flagships.' was not offered as a cut point"


def test_snap_end_reaches_a_sentence_across_a_tiny_gap():
    words = words_from([
        ("any", 0.0, 0.3), ("testing,", 0.3, 0.6),
        ("probably", 0.72, 1.1), ("the", 1.1, 1.3), ("flagships.", 1.3, 1.9),
        ("And", 1.98, 2.1), ("then", 2.1, 2.3),
    ])
    out = boundaries.snap_end(0.6, words, max_extend=6.0, max_truncate=1.5,
                              min_pause=0.18, tail=0.6, ceiling=60.0)
    assert out >= 1.9, f"stopped short of the sentence end (got {out})"


def test_cramped_cut_is_penalised_in_favour_of_nearby_silence():
    """Ending mid-phrase with no silence is worse than reaching a real pause."""
    words = words_from([
        ("this", 0.0, 0.3), ("is", 0.3, 0.5), ("a", 0.5, 0.6),
        ("passport", 0.6, 1.0), ("shaped", 1.0, 1.4), ("book", 1.4, 1.8),
        ("side,", 1.8, 2.2),
        ("with", 2.6, 2.9), ("more", 2.9, 3.2),
    ])
    # 0.6 is a hard splice: no gap, no punctuation. 2.2 has a 0.4s pause.
    out = boundaries.snap_end(0.6, words, max_extend=6.0, max_truncate=1.5,
                              min_pause=0.18, tail=0.6, ceiling=60.0)
    assert out > 2.2, f"stayed on the cramped cut (got {out})"


# --- asymmetric travel ---

def test_window_allows_more_travel_forward_than_backward():
    words = words_from([
        ("early.", 0.0, 0.5),
        ("middle", 3.0, 3.4), ("of", 3.4, 3.6), ("it", 3.6, 3.9),
        ("later.", 7.0, 7.6),
    ])
    # From 3.9: the sentence behind is 3.4s back, the one ahead 3.7s forward.
    out = boundaries.snap_end(3.9, words, max_extend=6.0, max_truncate=1.5,
                              min_pause=0.18, tail=0.6, ceiling=60.0)
    assert out > 3.9, "truncated backwards despite a reachable sentence ahead"


def test_max_truncate_bounds_backward_movement():
    words = words_from([
        ("done.", 0.0, 0.5),
        ("a", 5.0, 5.2), ("b", 5.2, 5.4), ("c", 5.4, 5.6),
    ])
    # The only sentence break is 5.1s behind, outside a 1.5s truncate budget.
    out = boundaries.snap_end(5.6, words, max_extend=6.0, max_truncate=1.5,
                              min_pause=0.18, tail=0.6, ceiling=60.0)
    assert out >= 5.6, f"reached back past the truncate budget (got {out})"


def test_max_shift_still_sets_both_directions():
    words = words_from([("a.", 0.0, 0.5), ("b", 2.0, 2.4), ("c.", 2.4, 3.0)])
    symmetric = boundaries.snap_end(2.4, words, max_shift=2.0, min_pause=0.18,
                                    tail=0.6, ceiling=60.0)
    assert isinstance(symmetric, float)


def test_length_clamp_does_not_cut_mid_word():
    """Hitting the duration cap must still land on a word boundary.

    Regression: the cap was applied as `start + max_seconds`, dropping the cut
    at an arbitrary instant and undoing the boundary search above it.
    """
    words = words_from([(f"w{i}", 10.0 + i * 0.37, 10.0 + (i + 1) * 0.37)
                        for i in range(80)])
    start, end = boundaries.snap_span(
        11.0, 40.0, words, max_extend=6.0, max_truncate=1.5, min_pause=0.18,
        lead_in=0.25, tail=0.6, media_duration=120.0, max_seconds=12.0,
    )
    assert end - start <= 12.0 + 1e-6
    straddled = [w for w in words if w["start"] < end < w["end"]]
    assert not straddled, f"cap cut inside {straddled}"


def test_minimum_length_extension_does_not_cut_mid_word():
    words = words_from([(f"w{i}", 10.0 + i * 0.37, 10.0 + (i + 1) * 0.37)
                        for i in range(80)])
    start, end = boundaries.snap_span(
        11.0, 13.0, words, max_extend=6.0, max_truncate=1.5, min_pause=0.18,
        lead_in=0.25, tail=0.6, media_duration=120.0, min_seconds=15.0,
    )
    straddled = [w for w in words if w["start"] < end < w["end"]]
    assert not straddled, f"minimum-length extension cut inside {straddled}"


def test_align_forward_moves_out_of_a_word():
    words = words_from([("hello", 1.0, 1.5), ("there", 1.5, 2.0)])
    assert boundaries.align_forward(words, 1.2) == 1.5
    assert boundaries.align_forward(words, 1.5) == 1.5     # already a boundary
    assert boundaries.align_forward(words, 9.0) == 9.0     # past the speech


def test_align_back_moves_out_of_a_word():
    words = words_from([("hello", 1.0, 1.5), ("there", 1.5, 2.0)])
    assert boundaries.align_back(words, 1.2) == 1.0
    assert boundaries.align_back(words, 1.5) == 1.5
    assert boundaries.align_back(words, 0.2) == 0.2
