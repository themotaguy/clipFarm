"""Finding where a topic finishes, not just where a sentence does."""
from __future__ import annotations

from app.core import context, topics


def words_from(text: str, start: float = 0.0, step: float = 0.3) -> list[dict]:
    return [
        {"word": tok, "start": round(start + i * step, 3),
         "end": round(start + (i + 1) * step, 3)}
        for i, tok in enumerate(text.split())
    ]


TALK = (
    "There are three moves that every serious system uses. "        # s0
    "The first is batching requests together. "                     # s1
    "It gives you ten times the throughput. "                       # s2
    "So let me talk about caching instead. "                        # s3  <- marker
    "Prefix caching reuses the attention state. "                   # s4
    "Users feel that difference immediately. "                      # s5
    "Now here is the uncomfortable part. "                          # s6  <- marker
    "Most teams just buy more hardware. "                           # s7
)


# --- discourse markers ---

def test_marker_opening_closes_the_previous_thought():
    sents = context.sentences(words_from(TALK))
    gaps = topics.marker_boundaries(sents)
    # "So let me talk about caching" is sentence 3, so the thought ended at gap 2.
    assert 2 in gaps
    # "Now here is the uncomfortable part" is sentence 6 -> gap 5.
    assert 5 in gaps


def test_markers_do_not_fire_mid_thought():
    sents = context.sentences(words_from(TALK))
    gaps = topics.marker_boundaries(sents)
    assert 0 not in gaps, "'The first is batching' continues the same thought"


def test_closing_marker_ends_a_thought_where_it_ends():
    sents = context.sentences(words_from(
        "That is why you measure cost per request. And then we moved on. "
    ))
    assert 0 in topics.marker_boundaries(sents)


def test_marker_detection_ignores_leading_punctuation():
    assert topics._starts_with_marker("— So, anyway.", topics.TOPIC_MARKERS)
    assert topics._starts_with_marker('"Now, listen."', topics.TOPIC_MARKERS)
    assert not topics._starts_with_marker("Batching helps.", topics.TOPIC_MARKERS)


# --- boundary times ---

def test_find_boundaries_returns_ordered_times():
    words = words_from(TALK)
    out = topics.find_boundaries(words)
    assert out == sorted(out)
    assert out, "expected at least one boundary"
    assert all(isinstance(t, float) for t in out)


def test_find_boundaries_always_closes_at_the_end_of_the_transcript():
    words = words_from("One thought only, with no markers at all. ")
    out = topics.find_boundaries(words)
    assert out and abs(out[-1] - words[-1]["end"]) < 1e-6


def test_find_boundaries_closes_a_single_sentence_at_its_end():
    out = topics.find_boundaries(words_from("Just one."))
    assert len(out) == 1
    assert topics.find_boundaries([]) == []


def test_next_boundary_picks_the_first_one_after():
    assert topics.next_boundary([10.0, 20.0, 30.0], 12.0) == 20.0
    assert topics.next_boundary([10.0, 20.0], 10.0) == 10.0
    assert topics.next_boundary([10.0], 50.0) is None
    assert topics.next_boundary([], 5.0) is None


# --- extending a clip to finish its thought ---

def test_complete_topic_runs_on_to_the_boundary():
    span = topics.complete_topic((0.0, 12.0), [18.0, 40.0],
                                 max_extend=45.0, max_seconds=180.0)
    assert span == (0.0, 18.0)


def test_complete_topic_declines_when_the_boundary_is_too_far():
    span = (0.0, 12.0)
    assert topics.complete_topic(span, [200.0], max_extend=45.0,
                                 max_seconds=180.0) == span


def test_complete_topic_declines_when_it_would_break_the_length_cap():
    span = (0.0, 100.0)
    assert topics.complete_topic(span, [140.0], max_extend=45.0,
                                 max_seconds=120.0) == span


def test_complete_topic_leaves_an_already_complete_span_alone():
    span = (0.0, 30.0)
    assert topics.complete_topic(span, [30.0], max_extend=45.0,
                                 max_seconds=180.0) == span
    assert topics.complete_topic(span, [], max_extend=45.0,
                                 max_seconds=180.0) == span


# --- semantic pass ---

def _fake_embed(groups: dict[str, list[float]]):
    """Embed by keyword, so 'topics' are trivially separable."""
    def embed(texts):
        out = []
        for text in texts:
            vec = [0.0, 0.0, 0.0]
            lowered = text.lower()
            for i, (key, _v) in enumerate(groups.items()):
                if key in lowered:
                    vec[i] = 1.0
            if not any(vec):
                vec = [0.3, 0.3, 0.3]
            out.append(vec)
        return out
    return embed


def test_semantic_pass_finds_a_seam_between_unrelated_blocks():
    text = (
        "Batching helps throughput. Batching groups requests. Batching is cheap. "
        "Batching scales well. "
        "Cameras are lovely. Cameras have sensors. Cameras need light. "
        "Cameras focus fast. "
    )
    sents = context.sentences(words_from(text))
    embed = _fake_embed({"batching": [1.0], "cameras": [1.0]})
    gaps = topics.semantic_boundaries(sents, embed, threshold=0.62)
    assert gaps, "expected a seam between the two subjects"
    # The seam sits at the switch, around gap index 3.
    assert any(abs(g - 3) <= 1 for g in gaps)


def test_semantic_pass_is_quiet_on_a_coherent_passage():
    text = ("Batching helps throughput. Batching groups requests. "
            "Batching is cheap. Batching scales well. Batching is the default. "
            "Batching wins. ")
    sents = context.sentences(words_from(text))
    embed = _fake_embed({"batching": [1.0]})
    assert topics.semantic_boundaries(sents, embed, threshold=0.62) == set()


def test_semantic_pass_needs_enough_sentences():
    sents = context.sentences(words_from("One. Two. "))
    assert topics.semantic_boundaries(sents, _fake_embed({}), threshold=0.62) == set()


def test_semantic_pass_survives_a_broken_embedder():
    sents = context.sentences(words_from(TALK))

    def boom(_texts):
        raise RuntimeError("ollama down")

    # Embeddings only sharpen the result, so a failure must not be fatal.
    assert topics.semantic_boundaries(sents, boom, threshold=0.62) == set()
    assert topics.find_boundaries(words_from(TALK), embed=boom)


def test_cosine_basics():
    assert abs(topics._cosine([1, 0], [1, 0]) - 1.0) < 1e-9
    assert abs(topics._cosine([1, 0], [0, 1])) < 1e-9
    assert topics._cosine([0, 0], [1, 1]) == 0.0


# --- pause signal ---

def test_pause_boundaries_fire_on_a_real_breath():
    sents = [
        {"text": "First point done.", "start": 0.0, "end": 2.0},
        {"text": "Second point starts.", "start": 2.6, "end": 4.0},   # 0.6s gap
        {"text": "Still the same point.", "start": 4.1, "end": 5.0},  # 0.1s gap
    ]
    gaps = topics.pause_boundaries(sents, min_pause=0.45)
    assert gaps == {0}


def test_pause_boundaries_respect_the_threshold():
    sents = [
        {"text": "A.", "start": 0.0, "end": 1.0},
        {"text": "B.", "start": 1.3, "end": 2.0},
    ]
    assert topics.pause_boundaries(sents, min_pause=0.45) == set()
    assert topics.pause_boundaries(sents, min_pause=0.25) == {0}


def test_find_boundaries_uses_pauses_without_any_markers():
    """No discourse markers anywhere, but the speaker still breathes."""
    words = (
        words_from("The camera has a new sensor.", start=0.0)
        + words_from("Battery life is much better.", start=5.0)
    )
    out = topics.find_boundaries(words)
    # The 3.2s gap between the two sentences is a boundary, plus the close.
    assert len(out) >= 2
