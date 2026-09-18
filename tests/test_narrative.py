"""Opening on the hook and landing the payoff."""
from __future__ import annotations

from app.core import context, narrative


def words_from(text: str, start: float = 0.0, step: float = 0.3) -> list[dict]:
    return [
        {"word": tok, "start": round(start + i * step, 3),
         "end": round(start + (i + 1) * step, 3)}
        for i, tok in enumerate(text.split())
    ]


def lines_from(texts: list[str], step: float = 4.0) -> list[dict]:
    return [
        {"n": i, "start": i * step, "end": (i + 1) * step - 0.2, "text": t}
        for i, t in enumerate(texts)
    ]


# --- promise / reveal language ---

def test_promise_detection():
    assert narrative.promises_more(
        "The new top of the lineup is the one more thing that we all saw coming."
    )
    assert narrative.promises_more("Wait until you see what happens next.")
    assert not narrative.promises_more("The camera has a 48 megapixel sensor.")


def test_reveal_detection():
    assert narrative.is_reveal("The first folding iPhone.")
    assert narrative.is_reveal("This is the iPhone Duo and it is wild.")
    assert narrative.is_reveal("Introducing the thing nobody expected.")
    assert not narrative.is_reveal("It costs a hundred dollars more.")


# --- locating the hook ---

LINES = lines_from([
    "So a hundred bucks more for a 16 or a 17e.",
    "That stuff is all on their site.",
    "But it's not the top of the lineup anymore.",
    "The first genuinely new form factor of the iPhone since 2007.",
    "The first folding iPhone.",
])


def test_hook_located_by_verbatim_text():
    hook, score = narrative.locate_hook(
        "The first genuinely new form factor of the iPhone since 2007.", None, LINES
    )
    assert hook is not None and hook["n"] == 3
    assert score > 0.9


def test_hook_line_is_used_when_it_agrees_with_the_text():
    hook, score = narrative.locate_hook("The first folding iPhone.", 4, LINES)
    assert hook["n"] == 4
    assert score > 0.9


def test_wrong_hook_line_is_overridden_by_the_text():
    """The model's index and its quote disagree often enough to matter."""
    hook, _score = narrative.locate_hook("The first folding iPhone.", 0, LINES)
    assert hook["n"] == 4, "a bad index moved the clip to the wrong place"


def test_unmatchable_hook_is_rejected():
    hook, score = narrative.locate_hook(
        "Something about entirely unrelated gardening equipment", None, LINES
    )
    assert hook is None
    assert score < 0.6


def test_blank_hook_is_rejected():
    assert narrative.locate_hook("", None, LINES) == (None, 0.0)
    assert narrative.locate_hook("anything", None, []) == (None, 0.0)


def test_hook_text_tolerates_the_marker_prefix():
    """Prompt lines are prefixed with '>', and the model echoes it back."""
    hook, _ = narrative.locate_hook(">The first folding iPhone.", None, LINES)
    assert hook["n"] == 4


# --- anchoring ---

def test_anchor_moves_the_start_onto_the_hook():
    hook = {"start": 40.0, "end": 44.0}
    start, end = narrative.anchor_to_hook(
        (10.0, 60.0), hook, run_up=1.0, max_trim=45.0, ceiling=180.0
    )
    assert abs(start - 39.0) < 1e-6, "should open just before the hook"
    assert end == 60.0


def test_anchor_respects_the_trim_cap():
    hook = {"start": 100.0, "end": 104.0}
    start, _end = narrative.anchor_to_hook(
        (10.0, 130.0), hook, run_up=1.0, max_trim=15.0, ceiling=180.0
    )
    assert abs(start - 25.0) < 1e-6, "trimmed further than the cap allows"


def test_anchor_extends_to_contain_a_hook_beyond_the_end():
    """The reported failure: the clip stopped 0.2s before its own hook."""
    hook = {"start": 238.1, "end": 246.5}
    start, end = narrative.anchor_to_hook(
        (194.9, 237.9), hook, run_up=1.0, max_trim=45.0, ceiling=180.0
    )
    assert end >= 246.5, "the hook is still outside the clip"
    assert start > 194.9, "should also have trimmed the preamble"


def test_anchor_pulls_back_to_include_an_earlier_hook():
    hook = {"start": 50.0, "end": 54.0}
    start, _end = narrative.anchor_to_hook(
        (60.0, 100.0), hook, run_up=1.0, max_trim=45.0, ceiling=180.0
    )
    assert start <= 50.0


def test_anchor_is_a_no_op_without_a_hook():
    span = (10.0, 40.0)
    assert narrative.anchor_to_hook(span, None, run_up=1.0, max_trim=15.0,
                                    ceiling=180.0) == span


def test_anchor_respects_the_duration_ceiling():
    hook = {"start": 10.0, "end": 400.0}
    start, end = narrative.anchor_to_hook(
        (10.0, 30.0), hook, run_up=1.0, max_trim=15.0, ceiling=60.0
    )
    assert end - start <= 60.0 + 1e-6


def test_anchor_refuses_to_collapse_a_span():
    hook = {"start": 39.9, "end": 40.0}
    span = (10.0, 40.0)
    out = narrative.anchor_to_hook(span, hook, run_up=0.0, max_trim=45.0,
                                   ceiling=180.0)
    assert out[1] - out[0] >= 1.0


# --- landing the payoff ---

BUILDUP = (
    "But it's not the top of the lineup anymore. "
    "The new top of the lineup is the one more thing that we all saw coming. "
    "The first genuinely new form factor of the iPhone since 2007. "
    "This is the iPhone Duo and it is genuinely interesting. "
)


def test_next_reveal_found_within_the_window():
    sents = context.sentences(words_from(BUILDUP))
    end = sents[1]["end"]
    reveal = narrative.next_reveal(sents, end, window=25.0)
    assert reveal is not None
    assert "new form factor" in reveal["text"]


def test_next_reveal_ignores_one_beyond_the_window():
    """A reveal much later in the video is a different moment, not this payoff."""
    text = ("The new top of the lineup is the one more thing we all saw coming. "
            + "Filler about storage tiers and pricing. " * 30
            + "This is the iPhone Duo. ")
    sents = context.sentences(words_from(text))
    end = sents[0]["end"]
    assert narrative.next_reveal(sents, end, window=5.0) is None
    # With a wide enough window it is found again.
    assert narrative.next_reveal(sents, end, window=10_000.0) is not None


def test_complete_reveal_extends_past_the_buildup():
    sents = context.sentences(words_from(BUILDUP))
    span = (sents[0]["start"], sents[1]["end"])
    start, end = narrative.complete_reveal(span, sents, window=25.0, ceiling=180.0)
    assert end > sents[2]["start"], "the reveal is still missing"


def test_complete_reveal_declines_without_a_reveal_nearby():
    """A promise with nothing resolving it must be left alone.

    Verified against real footage: "you'd hope it gets better year after year"
    is followed by unrelated talk about capacities.
    """
    text = ("Which you know, you'd hope it gets better year after year. "
            "But the capacities don't look all that different. "
            "Storage is the same as last time around. ")
    sents = context.sentences(words_from(text))
    span = (sents[0]["start"], sents[0]["end"])
    assert narrative.complete_reveal(span, sents, window=25.0,
                                     ceiling=180.0) == span


def test_complete_reveal_respects_the_ceiling():
    sents = context.sentences(words_from(BUILDUP))
    span = (sents[0]["start"], sents[1]["end"])
    out = narrative.complete_reveal(span, sents, window=25.0, ceiling=1.0)
    assert out == span, "extended past the duration ceiling"


def test_complete_reveal_no_op_on_empty_input():
    span = (10.0, 20.0)
    assert narrative.complete_reveal(span, [], window=25.0, ceiling=180.0) == span
