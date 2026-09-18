"""Stages 4 & 5 — candidate selection and LLM response handling (no Ollama calls)."""
from __future__ import annotations

import config
from app.core import scoring


# --- JSON parsing of model output ---

def test_parses_clean_json():
    assert scoring._parse_json('{"title": "Hi", "virality_score": 80}')["title"] == "Hi"


def test_parses_json_wrapped_in_prose_and_markdown():
    raw = 'Sure! Here you go:\n```json\n{"title": "Hi"}\n```\nHope that helps.'
    assert scoring._parse_json(raw) == {"title": "Hi"}


def test_rejects_non_object_json():
    assert scoring._parse_json('["a", "b"]') is None
    assert scoring._parse_json('"just a string"') is None


def test_rejects_unparseable_and_empty():
    assert scoring._parse_json("not json at all") is None
    assert scoring._parse_json("") is None
    assert scoring._parse_json(None) is None


# --- field coercion ---

def test_clamp_bounds_and_defaults():
    assert scoring._clamp(150, 0, 100, default=50) == 100
    assert scoring._clamp(-10, 0, 100, default=50) == 0
    assert scoring._clamp(None, 0, 100, default=50) == 50
    assert scoring._clamp("garbage", 0, 100, default=50) == 50
    assert scoring._clamp("77", 0, 100, default=50) == 77


def test_clamp_index_always_returns_int():
    # Models sometimes emit floats or numeric strings for line numbers; these
    # are used as list subscripts, so a float would raise TypeError.
    for value in (2, 2.5, "2", "2.7", None, "abc", -5, 999, float("nan")):
        result = scoring._clamp_index(value, 0, 7, default=0)
        assert isinstance(result, int), f"{value!r} -> {result!r}"
        assert 0 <= result <= 7


def test_tags_normalised_from_list_and_string():
    assert scoring._as_tags(["AI", "#ML", " Growth "]) == ["ai", "ml", "growth"]
    assert scoring._as_tags("ai, ml; growth") == ["ai", "ml", "growth"]
    assert scoring._as_tags(None) == []
    assert len(scoring._as_tags([f"t{i}" for i in range(20)])) == 5


def test_title_stripped_of_quotes_and_collapsed():
    assert scoring._clean_title('  "A   great   clip"  ') == "A great clip"
    assert len(scoring._clean_title("x" * 200)) == 80


# --- span selection ---

def test_dedupe_keeps_first_and_drops_overlapping():
    items = [
        {"start": 0.0, "end": 40.0, "tag": "keep"},
        {"start": 5.0, "end": 42.0, "tag": "drop"},     # heavy overlap
        {"start": 100.0, "end": 140.0, "tag": "keep2"},
    ]
    kept = scoring.dedupe_spans(items, max_overlap=0.35)
    assert [i["tag"] for i in kept] == ["keep", "keep2"]


def test_dedupe_allows_light_overlap():
    items = [
        {"start": 0.0, "end": 40.0},
        {"start": 38.0, "end": 78.0},   # 2s of 40s shared -> 5%
    ]
    assert len(scoring.dedupe_spans(items, max_overlap=0.35)) == 2


def test_dedupe_empty_input():
    assert scoring.dedupe_spans([]) == []


# --- prompt line construction ---

def test_build_lines_selects_overlapping_segments_only(transcript):
    lines = scoring.build_lines(transcript, 10.0, 25.0)
    assert lines
    assert all(l["end"] > 10.0 and l["start"] < 25.0 for l in lines)
    # Numbering must be dense and zero-based, since the model replies with indices.
    assert [l["n"] for l in lines] == list(range(len(lines)))


def test_build_lines_empty_when_range_has_no_speech(transcript):
    assert scoring.build_lines(transcript, 10_000.0, 10_050.0) == []


def test_format_lines_includes_index_and_timestamps():
    out = scoring._format_lines([{"n": 0, "start": 1.5, "end": 4.25, "text": "Hello."}])
    assert "[0]" in out and "1.5s" in out and "Hello." in out


# --- bound refinement ---

def test_refine_bounds_honours_model_choice(transcript):
    lines = scoring.build_lines(transcript, 0.0, 1000.0)
    cand = {"start": 0.0, "end": 45.0, "retrieval_score": 0.5}
    start, end = scoring._refine_bounds(lines, {"start_line": 2, "end_line": 8}, cand)
    assert start == lines[2]["start"]
    assert end >= start


def test_refine_bounds_extends_short_selection(transcript):
    lines = scoring.build_lines(transcript, 0.0, 1000.0)
    cand = {"start": 0.0, "end": 45.0, "retrieval_score": 0.5}
    # A single line is far below CLIP_MIN_SECONDS and must be padded out.
    start, end = scoring._refine_bounds(lines, {"start_line": 3, "end_line": 3}, cand)
    assert end - start >= config.CLIP_MIN_SECONDS - 0.01


def test_refine_bounds_trims_long_selection(transcript):
    lines = scoring.build_lines(transcript, 0.0, 1000.0)
    cand = {"start": 0.0, "end": 45.0, "retrieval_score": 0.5}
    start, end = scoring._refine_bounds(
        lines, {"start_line": 0, "end_line": len(lines) - 1}, cand
    )
    assert end - start <= config.CLIP_MAX_SECONDS + 0.01


def test_refine_bounds_swaps_reversed_lines(transcript):
    lines = scoring.build_lines(transcript, 0.0, 1000.0)
    cand = {"start": 0.0, "end": 45.0, "retrieval_score": 0.5}
    start, end = scoring._refine_bounds(lines, {"start_line": 7, "end_line": 2}, cand)
    assert end > start


def test_refine_bounds_survives_garbage_line_numbers(transcript):
    lines = scoring.build_lines(transcript, 0.0, 1000.0)
    cand = {"start": 0.0, "end": 45.0, "retrieval_score": 0.5}
    for payload in (
        {"start_line": 2.5, "end_line": 6.5},
        {"start_line": "2", "end_line": "6"},
        {"start_line": None, "end_line": None},
        {},
        {"start_line": -99, "end_line": 9999},
    ):
        start, end = scoring._refine_bounds(lines, payload, cand)
        assert end > start, payload


# --- fallback path ---

def test_fallback_is_renderable(transcript):
    cand = {
        "start": 10.0, "end": 40.0, "text": "Some retrieved text here.",
        "retrieval_score": 0.62, "relevance": 0.6, "match_count": 2,
        "matched_queries": [], "chunk_id": "chunk-00001",
    }
    out = scoring._fallback(cand, "ollama down")
    # The pipeline reads all of these when building a clip record.
    for key in ("title", "hook", "summary", "reason", "tags",
                "virality_score", "breakdown", "final_score", "scored_by"):
        assert key in out
    assert out["scored_by"] == "retrieval-fallback"
    assert out["duration"] == 30.0


def test_score_candidate_falls_back_without_lines():
    cand = {"start": 0.0, "end": 30.0, "text": "t", "retrieval_score": 0.5}
    out = scoring.score_candidate(cand, [])
    assert out["scored_by"] == "retrieval-fallback"


def test_score_candidate_uses_fallback_when_model_returns_junk(monkeypatch, transcript):
    monkeypatch.setattr(scoring, "_invoke", lambda prompt: "I'm afraid I can't do that.")
    cand = {"start": 0.0, "end": 45.0, "text": "t", "retrieval_score": 0.5}
    lines = scoring.build_lines(transcript, 0.0, 1000.0)
    out = scoring.score_candidate(cand, lines)
    assert out["scored_by"] == "retrieval-fallback"
    assert "parseable JSON" in out["reason"]


def test_score_candidate_builds_full_record(monkeypatch, transcript):
    payload = (
        '{"title": "Sports car", "hook": "You are paying for a sports car.",'
        ' "summary": "On wasted GPU capacity.", "hook_strength": 9,'
        ' "emotional_impact": 6, "standalone_clarity": 8, "shareability": 7,'
        ' "virality_score": 85, "start_line": 1, "end_line": 6,'
        ' "tags": ["ai", "performance"], "reason": "Strong hook."}'
    )
    monkeypatch.setattr(scoring, "_invoke", lambda prompt: payload)
    cand = {"start": 0.0, "end": 45.0, "text": "t", "retrieval_score": 0.8}
    lines = scoring.build_lines(transcript, 0.0, 1000.0)
    out = scoring.score_candidate(cand, lines)

    assert out["title"] == "Sports car"
    assert out["virality_score"] == 85
    assert out["breakdown"]["hook_strength"] == 9
    assert out["tags"] == ["ai", "performance"]
    assert out["scored_by"] == config.OLLAMA_MODEL
    # final = 0.75 * 85 + 25 * 0.8
    assert abs(out["final_score"] - (0.75 * 85 + 25 * 0.8)) < 0.01
    assert out["duration"] == round(out["end"] - out["start"], 3)


def test_score_candidate_retries_once_then_succeeds(monkeypatch, transcript):
    calls = []

    def flaky(prompt):
        calls.append(prompt)
        if len(calls) == 1:
            return "no json here"
        return '{"title": "Second try", "virality_score": 70, "start_line": 0, "end_line": 4}'

    monkeypatch.setattr(scoring, "_invoke", flaky)
    cand = {"start": 0.0, "end": 45.0, "text": "t", "retrieval_score": 0.5}
    out = scoring.score_candidate(cand, scoring.build_lines(transcript, 0.0, 1000.0))
    assert len(calls) == 2
    assert out["title"] == "Second try"


# --- retrieval ---

class _FakeIndex:
    """Stands in for StreamingIndex.multi_query."""

    def __init__(self, hits):
        self._hits = hits

    def multi_query(self, queries, k=8):
        return self._hits


def test_retrieve_candidates_ranks_and_dedupes():
    hits = {
        "chunk-0": {
            "id": "chunk-0", "text": "a", "relevance": 0.9, "match_count": 4,
            "matched_queries": [{"query": q} for q in "abcd"],
            "metadata": {"start": 0.0, "end": 40.0},
        },
        "chunk-1": {   # near-duplicate span of chunk-0, weaker -> dropped
            "id": "chunk-1", "text": "b", "relevance": 0.5, "match_count": 1,
            "matched_queries": [{"query": "a"}],
            "metadata": {"start": 2.0, "end": 41.0},
        },
        "chunk-2": {
            "id": "chunk-2", "text": "c", "relevance": 0.7, "match_count": 2,
            "matched_queries": [{"query": "a"}, {"query": "b"}],
            "metadata": {"start": 200.0, "end": 240.0},
        },
    }
    out = scoring.retrieve_candidates(_FakeIndex(hits), limit=10)
    ids = [c["chunk_id"] for c in out]
    assert ids == ["chunk-0", "chunk-2"]
    assert out[0]["retrieval_score"] >= out[1]["retrieval_score"]
    assert out[0]["match_count"] == 4


def test_retrieve_candidates_respects_limit():
    hits = {
        f"chunk-{i}": {
            "id": f"chunk-{i}", "text": "x", "relevance": 0.5 + i / 100,
            "match_count": 1, "matched_queries": [{"query": "a"}],
            "metadata": {"start": i * 100.0, "end": i * 100.0 + 40},
        }
        for i in range(10)
    }
    assert len(scoring.retrieve_candidates(_FakeIndex(hits), limit=3)) == 3


def test_retrieve_candidates_empty_index():
    assert scoring.retrieve_candidates(_FakeIndex({})) == []


def test_probe_panel_is_non_trivial():
    assert len(scoring.VIRALITY_PROBES) >= 5
    assert len(set(scoring.VIRALITY_PROBES)) == len(scoring.VIRALITY_PROBES)


# --- stitching: the optional setup span ---

def _lines(transcript):
    return scoring.build_lines(transcript, 0.0, 10_000.0)


def test_no_setup_span_when_the_model_declines(transcript):
    lines = _lines(transcript)
    assert scoring._setup_span(lines, {}, 20.0) is None
    assert scoring._setup_span(
        lines, {"setup_start_line": None, "setup_end_line": None}, 20.0
    ) is None


def test_setup_span_is_taken_from_earlier_lines(transcript):
    lines = _lines(transcript)
    main_start = lines[6]["start"]
    span = scoring._setup_span(
        lines, {"setup_start_line": 1, "setup_end_line": 2}, main_start
    )
    assert span is not None
    assert span[0] >= 0.0
    assert span[1] <= main_start + 1e-9, "setup must end before the payoff"


def test_setup_span_is_trimmed_back_to_the_payoff(transcript):
    lines = _lines(transcript)
    main_start = lines[4]["start"]
    # Ask for a range that runs past the payoff; it must be clipped.
    span = scoring._setup_span(
        lines, {"setup_start_line": 2, "setup_end_line": 9}, main_start
    )
    assert span is not None
    assert span[1] <= main_start + 1e-9


def test_setup_span_dropped_when_nothing_useful_remains(transcript):
    lines = _lines(transcript)
    # A "setup" that starts after the payoff leaves no usable audio.
    assert scoring._setup_span(
        lines, {"setup_start_line": 8, "setup_end_line": 9}, lines[0]["start"]
    ) is None


def test_setup_span_respects_its_own_length_cap(transcript, monkeypatch):
    monkeypatch.setattr(config, "SETUP_MAX_SECONDS", 4.0)
    lines = _lines(transcript)
    span = scoring._setup_span(
        lines, {"setup_start_line": 0, "setup_end_line": 8}, lines[9]["start"]
    )
    assert span is not None
    assert span[1] - span[0] <= 4.0 + 1e-9


def test_setup_span_disabled_by_config(transcript, monkeypatch):
    monkeypatch.setattr(config, "STITCH_SETUP", False)
    lines = _lines(transcript)
    assert scoring._setup_span(
        lines, {"setup_start_line": 0, "setup_end_line": 2}, lines[5]["start"]
    ) is None


def test_compose_spans_single_when_no_setup(transcript):
    words = [w for s in transcript for w in s["words"]]
    spans = scoring._compose_spans((20.0, 45.0), None, words, 300.0)
    assert len(spans) == 1


def test_compose_spans_puts_setup_first_and_keeps_them_disjoint(transcript):
    words = [w for s in transcript for w in s["words"]]
    spans = scoring._compose_spans((25.0, 45.0), (4.0, 12.0), words, 300.0)
    assert len(spans) == 2
    assert spans[0][1] <= spans[1][0], "spans overlap"
    assert spans[0][0] < spans[1][0]


def test_compose_spans_total_stays_inside_the_clip_cap(transcript, monkeypatch):
    monkeypatch.setattr(config, "CLIP_MAX_SECONDS", 30.0)
    words = [w for s in transcript for w in s["words"]]
    spans = scoring._compose_spans((20.0, 45.0), (0.0, 18.0), words, 300.0)
    total = sum(e - s for s, e in spans)
    assert total <= 30.0 + 1e-6


def test_compose_spans_fuses_a_setup_that_abuts_the_payoff(transcript):
    words = [w for s in transcript for w in s["words"]]
    spans = scoring._compose_spans((20.0, 40.0), (10.0, 20.0), words, 300.0)
    # Contiguous ranges are one continuous cut, not a "stitch".
    assert len(spans) == 1


def test_score_candidate_reports_spans_and_stitched_flag(monkeypatch, transcript):
    payload = (
        '{"title": "Ctx", "virality_score": 80, "start_line": 6, "end_line": 9,'
        ' "setup_start_line": 0, "setup_end_line": 1}'
    )
    monkeypatch.setattr(scoring, "_invoke", lambda prompt: payload)
    cand = {"start": transcript[6]["start"], "end": transcript[9]["end"],
            "text": "t", "retrieval_score": 0.6}
    words = [w for s in transcript for w in s["words"]]
    out = scoring.score_candidate(cand, _lines(transcript),
                                  words=words, media_duration=300.0)

    assert out["spans"], "spans missing"
    assert out["stitched"] is (len(out["spans"]) > 1)
    # duration is the sum of the spans, not the envelope.
    assert abs(out["duration"] - sum(s["end"] - s["start"] for s in out["spans"])) < 0.01
    # start/end describe the payoff, so dedupe still works on the main moment.
    assert out["end"] == out["spans"][-1]["end"]


def test_score_candidate_single_span_without_setup(monkeypatch, transcript):
    payload = '{"title": "Solo", "virality_score": 70, "start_line": 2, "end_line": 7}'
    monkeypatch.setattr(scoring, "_invoke", lambda prompt: payload)
    cand = {"start": 10.0, "end": 30.0, "text": "t", "retrieval_score": 0.5}
    words = [w for s in transcript for w in s["words"]]
    out = scoring.score_candidate(cand, _lines(transcript),
                                  words=words, media_duration=300.0)
    assert out["stitched"] is False
    assert len(out["spans"]) == 1


def test_fallback_still_reports_a_span(transcript):
    cand = {"start": 10.0, "end": 40.0, "text": "x", "retrieval_score": 0.5}
    words = [w for s in transcript for w in s["words"]]
    out = scoring._fallback(cand, "ollama down", words=words, media_duration=300.0)
    assert len(out["spans"]) == 1
    assert out["stitched"] is False
    assert out["duration"] > 0


def test_refine_bounds_defaults_to_the_payoff_not_the_whole_window(transcript):
    """With a padded context window, omitted line numbers must not widen the clip."""
    lines = _lines(transcript)
    cand = {"start": lines[5]["start"], "end": lines[7]["end"], "retrieval_score": 0.5}
    start, end = scoring._refine_bounds(lines, {}, cand, (5, 7))
    assert start >= lines[4]["start"], "clip reached back to the window start"


def test_format_lines_marks_the_retrieved_moment():
    lines = [
        {"n": 0, "start": 0.0, "end": 1.0, "text": "context"},
        {"n": 1, "start": 1.0, "end": 2.0, "text": "payoff"},
    ]
    out = scoring._format_lines(lines, [1])
    assert ">[1]" in out
    assert " [0]" in out


# --- duration spread ---

def test_duration_buckets_cover_the_whole_range():
    buckets = scoring.duration_buckets(5, 15.0, 180.0)
    assert len(buckets) == 5
    assert buckets[0][0] == 15.0
    assert buckets[-1][1] == 180.0
    # Contiguous, ascending, no gaps.
    for (a_lo, a_hi), (b_lo, b_hi) in zip(buckets, buckets[1:]):
        assert a_hi <= b_lo + 0.01
        assert a_lo < a_hi and b_lo < b_hi


def test_duration_buckets_are_finer_at_the_short_end():
    buckets = scoring.duration_buckets(5, 15.0, 180.0)
    first = buckets[0][1] - buckets[0][0]
    last = buckets[-1][1] - buckets[-1][0]
    assert first < last, "short clips need finer granularity than long ones"


def test_duration_buckets_degenerate_cases():
    assert scoring.duration_buckets(1, 15.0, 180.0) == [(15.0, 180.0)]
    assert scoring.duration_buckets(3, 60.0, 60.0) == [(60.0, 60.0)]
    assert len(scoring.duration_buckets(0, 15.0, 180.0)) == 1


def _clip(duration, score):
    return {"duration": duration, "final_score": score,
            "start": 0.0, "end": duration}


def test_spread_picks_from_different_length_buckets():
    # Score order alone would take the four ~20s clips.
    clips = [
        _clip(20.0, 90), _clip(21.0, 89), _clip(22.0, 88), _clip(23.0, 87),
        _clip(60.0, 70), _clip(150.0, 60),
    ]
    out = scoring.spread_by_duration(clips, 4, min_seconds=15.0, max_seconds=180.0)
    durations = sorted(c["duration"] for c in out)
    assert len(out) == 4
    assert max(durations) >= 60.0, f"no long clip selected: {durations}"
    assert len(set(durations)) == 4


def test_spread_keeps_the_result_in_score_order():
    clips = [_clip(20.0, 90), _clip(60.0, 70), _clip(150.0, 60)]
    out = scoring.spread_by_duration(clips, 3, min_seconds=15.0, max_seconds=180.0)
    scores = [c["final_score"] for c in out]
    assert scores == sorted(scores, reverse=True)


def test_spread_backfills_when_buckets_are_empty():
    # Everything is short, so most buckets cannot be filled.
    clips = [_clip(16.0 + i, 90 - i) for i in range(6)]
    out = scoring.spread_by_duration(clips, 5, min_seconds=15.0, max_seconds=180.0)
    assert len(out) == 5


def test_spread_handles_fewer_clips_than_slots():
    out = scoring.spread_by_duration([_clip(20.0, 90)], 5,
                                     min_seconds=15.0, max_seconds=180.0)
    assert len(out) == 1


def test_spread_puts_overlong_clips_in_the_top_bucket():
    clips = [_clip(20.0, 50), _clip(250.0, 99)]
    out = scoring.spread_by_duration(clips, 2, min_seconds=15.0, max_seconds=180.0)
    assert len(out) == 2


def test_spread_empty_input():
    assert scoring.spread_by_duration([], 5) == []


def test_trimmed_setup_span_does_not_start_mid_word(transcript, monkeypatch):
    """Fitting a setup span into the remaining budget must respect words.

    Regression: the trim was `end - budget`, an arbitrary instant that could
    land inside a word and bypass all the boundary logic.
    """
    monkeypatch.setattr(config, "CLIP_MAX_SECONDS", 30.0)
    words = [w for s in transcript for w in s["words"]]
    # A long setup plus a long payoff, so the setup has to be trimmed hard.
    spans = scoring._compose_spans((28.0, 50.0), (0.0, 25.0), words, 300.0,
                                   (15.0, 30.0))
    for start, end in spans:
        for edge in (start, end):
            straddling = [
                w for w in words
                if float(w["start"]) < edge < float(w["end"])
            ]
            assert not straddling, f"edge {edge} lands inside {straddling}"


# --- critic pass ---

def _critic_lines(transcript):
    return scoring.build_lines(transcript, 0.0, 10_000.0)


def test_critique_is_skipped_without_lines(transcript):
    assert scoring.critique_clip({"start": 0.0, "end": 10.0}, []) is None


def test_happy_critique_changes_nothing(transcript):
    lines = _critic_lines(transcript)
    clip = {"start": lines[2]["start"], "end": lines[6]["end"],
            "hook": lines[2]["text"], "spans": [{"start": 0.0, "end": 1.0}]}
    out = scoring.apply_critique(
        clip,
        {"opens_on_hook": True, "contains_payoff": True,
         "has_housekeeping": False, "verdict": "good cut"},
        lines,
    )
    assert out["critique"]["applied"] is False
    assert out["spans"] == clip["spans"], "a passing review must not recut"


def test_critique_rejected_when_it_would_lose_the_hook(transcript):
    """Observed: the critic reports a missing payoff, then proposes bounds
    that still miss it. Adopting that would undo the reveal completion."""
    lines = _critic_lines(transcript)
    words = [w for s in transcript for w in s["words"]]
    hook = lines[8]
    clip = {"start": lines[2]["start"], "end": hook["end"],
            "hook": hook["text"],
            "spans": [{"start": lines[2]["start"], "end": hook["end"]}],
            "duration": hook["end"] - lines[2]["start"]}
    out = scoring.apply_critique(
        clip,
        {"opens_on_hook": False, "contains_payoff": False,
         "has_housekeeping": False,
         # A recut that drops the hook entirely.
         "suggested_start_line": 0, "suggested_end_line": 4,
         "verdict": "needs work"},
        lines, words=words, media_duration=1000.0,
    )
    assert out["critique"]["applied"] is False
    assert out["spans"] == clip["spans"]


def test_an_adopted_recut_is_always_a_legal_clip(transcript, monkeypatch):
    """A one-line suggestion is grown to the minimum, not shipped as a scrap."""
    monkeypatch.setattr(config, "CLIP_MIN_SECONDS", 30.0)
    lines = _critic_lines(transcript)
    clip = {"start": lines[0]["start"], "end": lines[9]["end"], "hook": "",
            "spans": [{"start": lines[0]["start"], "end": lines[9]["end"]}],
            "duration": 40.0}
    out = scoring.apply_critique(
        clip,
        {"opens_on_hook": False, "contains_payoff": True,
         "has_housekeeping": False,
         "suggested_start_line": 0, "suggested_end_line": 0,
         "verdict": "trim it"},
        lines, words=[w for s in transcript for w in s["words"]],
        media_duration=1000.0,
    )
    if out["critique"]["applied"]:
        assert out["duration"] >= config.CLIP_MIN_SECONDS - 0.5
        assert out["duration"] <= config.CLIP_MAX_SECONDS + 0.5


def test_critique_rejected_when_it_lands_in_an_advert(transcript):
    lines = _critic_lines(transcript)
    words = [w for s in transcript for w in s["words"]]
    advert = [{"start": lines[0]["start"], "end": lines[5]["end"]}]
    clip = {"start": lines[6]["start"], "end": lines[10]["end"], "hook": "",
            "spans": [{"start": lines[6]["start"], "end": lines[10]["end"]}],
            "duration": 30.0}
    out = scoring.apply_critique(
        clip,
        {"opens_on_hook": False, "contains_payoff": True,
         "has_housekeeping": False,
         "suggested_start_line": 0, "suggested_end_line": 3,
         "verdict": "move it"},
        lines, words=words, media_duration=1000.0, ad_spans=advert,
    )
    assert out["critique"]["applied"] is False


def test_critique_absent_leaves_the_clip_alone():
    clip = {"start": 1.0, "end": 2.0, "spans": [{"start": 1.0, "end": 2.0}]}
    assert scoring.apply_critique(clip, None, []) == clip
