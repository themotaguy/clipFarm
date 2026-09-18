"""Detecting clips that open on an unexplained reference."""
from __future__ import annotations

from app.core import context


def words_from(text: str, start: float = 0.0, step: float = 0.3) -> list[dict]:
    return [
        {"word": tok, "start": round(start + i * step, 3),
         "end": round(start + (i + 1) * step, 3)}
        for i, tok in enumerate(text.split())
    ]


# --- sentence splitting ---

def test_sentences_split_on_terminal_punctuation():
    words = words_from("One two three. Four five! Six seven?")
    out = context.sentences(words)
    assert len(out) == 3
    assert out[0]["text"] == "One two three."
    assert out[0]["start"] == 0.0
    assert out[2]["text"].endswith("?")


def test_sentences_keeps_a_trailing_fragment():
    out = context.sentences(words_from("A complete one. And a dangling"))
    assert len(out) == 2
    assert out[1]["text"] == "And a dangling"


def test_sentences_empty_input():
    assert context.sentences([]) == []


# --- reference detection ---

def test_detects_ordinal_enumeration_and_recovers_the_noun():
    ref = context.dangling_reference("The third move is simply not calling the model.")
    assert ref is not None
    assert ref["kind"] == "enumeration"
    assert ref["term"] == "move"


def test_detects_other_ordinals():
    for opening in (
        "The second option is caching.",
        "The last thing to try is routing.",
        "The other approach is batching.",
    ):
        assert context.dangling_reference(opening) is not None, opening


def test_detects_demonstrative_with_a_noun():
    ref = context.dangling_reference("That number is what kills your margin.")
    assert ref is not None
    assert ref["term"] == "number"


def test_detects_bare_pronoun_openings():
    for opening in ("It works, for a while.", "They will not tell you why.",
                    "This is not a joke."):
        assert context.dangling_reference(opening) is not None, opening


def test_detects_explicit_backreferences():
    ref = context.dangling_reference("As I said, the bill always arrives.")
    assert ref is not None
    assert ref["kind"] == "backreference"


def test_self_contained_openings_are_not_flagged():
    for opening in (
        "Here is the thing nobody tells you about inference cost.",
        "You are paying for a sports car and driving it in a parking lot.",
        "Measure your cost per request on day one.",
        "Continuous batching gives you ten times throughput.",
    ):
        assert context.dangling_reference(opening) is None, opening


def test_empty_text_is_not_flagged():
    assert context.dangling_reference("") is None
    assert context.dangling_reference("   ") is None


# --- setup selection ---

TALK = (
    "There are three moves that every serious system uses. "          # 0
    "The first move is batching requests together. "                  # 1
    "The second move is caching the attention state. "                # 2
    "The third move is simply not calling the model at all. "         # 3
    "A small classifier decides whether it is needed. "               # 4
)


def test_suggest_setup_finds_the_sentence_that_introduced_the_term():
    words = words_from(TALK)
    sents = context.sentences(words)
    main_start = sents[3]["start"]          # "The third move ..."

    span = context.suggest_setup(words, main_start, lookback=60.0, max_seconds=20.0)
    assert span is not None
    start, end = span
    assert end <= main_start + 0.01, "setup must end before the clip"
    # It should reach back to a sentence that actually mentions "moves".
    covered = [s for s in sents if s["start"] >= start - 0.01 and s["end"] <= end + 0.01]
    assert any("moves" in s["text"] or "move" in s["text"] for s in covered)


def test_suggest_setup_returns_none_for_a_self_contained_clip():
    words = words_from(
        "Here is the thing nobody tells you. "
        "You are paying for a sports car and driving it in a parking lot. "
    )
    sents = context.sentences(words)
    assert context.suggest_setup(
        words, sents[1]["start"], lookback=60.0, max_seconds=20.0
    ) is None


def test_suggest_setup_respects_the_length_cap():
    words = words_from(TALK)
    sents = context.sentences(words)
    span = context.suggest_setup(
        words, sents[3]["start"], lookback=60.0, max_seconds=1.5
    )
    if span is not None:
        assert span[1] - span[0] <= 1.5 + 1e-6


def test_suggest_setup_honours_the_lookback_window():
    words = words_from(TALK)
    sents = context.sentences(words)
    # Nothing is visible within a 0.1s lookback.
    assert context.suggest_setup(
        words, sents[3]["start"], lookback=0.1, max_seconds=20.0
    ) is None


def test_suggest_setup_none_at_the_very_start_of_the_media():
    words = words_from(TALK)
    assert context.suggest_setup(words, 0.0, lookback=60.0, max_seconds=20.0) is None


def test_suggest_setup_none_without_words():
    assert context.suggest_setup([], 30.0, lookback=60.0, max_seconds=20.0) is None


def test_suggest_setup_falls_back_to_the_preceding_sentence_for_a_pronoun():
    text = ("Hardware is available and budgets are generous. "
            "It always works, right up until it does not. ")
    words = words_from(text)
    sents = context.sentences(words)
    span = context.suggest_setup(words, sents[1]["start"], lookback=60.0,
                                 max_seconds=20.0)
    assert span is not None
    assert span[0] <= sents[0]["start"] + 0.01
