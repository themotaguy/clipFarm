"""Shaping a clip so it opens on its hook and lands its payoff.

Two failures motivate this, both seen on a real video:

**The clip does not start at its hook.** The scorer names the single most
scroll-stopping sentence in its `hook` field, and then routinely sets in/out
points that begin 6-43 seconds earlier — on price admin, or a subscribe ask.
Nothing in those opening seconds makes anyone stay.

**The clip stops just before the reveal.** One clip ended on "The new top of
the lineup is the one more thing that we all saw coming." — the entire buildup,
with the payoff ("The first folding iPhone.") starting 0.2s after the cut. In
that same clip the scorer's own `hook` was the reveal sentence, sitting outside
the span it had chosen.

So the model already produces what is needed; it just cannot be trusted to act
on it. Both rules are enforced here instead.
"""

from __future__ import annotations

import difflib
from typing import Any, Sequence

#: Openings that promise something the clip has not delivered yet.
PROMISE_MARKERS = (
    "one more thing", "saw coming", "coming up", "wait until", "wait for it",
    "you'll see", "you will see", "i'll show you", "let me show you",
    "the best part", "but first", "stay tuned", "here's where it gets",
    "heres where it gets", "which brings me", "you won't believe",
    "the craziest part", "and then it happened",
)

#: Sentences that deliver a reveal. Deliberately a small, high-precision set.
REVEAL_MARKERS = (
    "this is the", "this is it", "the first ", "introducing", "meet the",
    "say hello to", "it's called", "its called", "here it is",
    "and that is the", "turns out",
)


def _text(sentence: dict[str, Any]) -> str:
    return (sentence.get("text") or "").strip()


def _has(text: str, markers: Sequence[str]) -> bool:
    lowered = text.lower()
    return any(m in lowered for m in markers)


def promises_more(text: str) -> bool:
    """Whether `text` sets up something it does not itself resolve."""
    return _has(text, PROMISE_MARKERS)


def is_reveal(text: str) -> bool:
    """Whether `text` reads like the payoff of a buildup."""
    return _has(text, REVEAL_MARKERS)


# --- locating the hook ---------------------------------------------------

def locate_hook(
    hook_text: str,
    hook_line: Any,
    lines: Sequence[dict[str, Any]],
    *,
    threshold: float = 0.6,
) -> tuple[dict[str, Any] | None, float]:
    """Find the prompt line the scorer's `hook` refers to, with a confidence.

    Prefers the line number the model reported, but only when its text really
    does resemble the quoted hook — the two disagree often enough that trusting
    the index blindly would move clips to the wrong place. Falls back to the
    best fuzzy match over all lines.

    The confidence matters: how far a clip may be trimmed to reach the hook
    depends on how sure we are of where the hook is.
    """
    hook = (hook_text or "").strip().lstrip(">").strip()
    if not hook or not lines:
        return None, 0.0

    def ratio(candidate: str) -> float:
        return difflib.SequenceMatcher(
            None, hook[:80].lower(), (candidate or "")[:80].lower()
        ).ratio()

    # The model's own index, if it is plausible and consistent with the text.
    try:
        index = int(round(float(hook_line)))
    except (TypeError, ValueError):
        index = None
    if index is not None and 0 <= index < len(lines):
        stated = lines[index]
        stated_score = ratio(stated.get("text", ""))
        if stated_score >= threshold:
            return stated, stated_score

    best, score = None, 0.0
    for line in lines:
        value = ratio(line.get("text", ""))
        if value > score:
            best, score = line, value
    if best is None or score < threshold:
        return None, score
    return best, score


def anchor_to_hook(
    span: tuple[float, float],
    hook: dict[str, Any] | None,
    *,
    run_up: float,
    max_trim: float,
    ceiling: float,
) -> tuple[float, float]:
    """Move a span so it opens on the hook, and always contains it.

    The trim is capped: a mis-matched hook must not be able to gut a clip.
    """
    start, end = span
    if not hook:
        return span

    hook_start = float(hook["start"])
    hook_end = float(hook["end"])

    # Containment first — a hook outside the span means the clip stops short of
    # its own payoff, which is exactly the reveal-cut-off failure.
    if hook_end > end:
        end = min(hook_end, start + ceiling)
    if hook_start < start:
        start = hook_start

    # Then open on it, skipping no more than `max_trim` of preamble.
    lead = hook_start - start
    if lead > run_up:
        start = start + min(lead - run_up, max_trim)

    if end - start < 1.0:
        return span
    return (round(start, 3), round(end, 3))


# --- landing the payoff --------------------------------------------------

def next_reveal(
    sentences: Sequence[dict[str, Any]],
    after: float,
    *,
    window: float,
) -> dict[str, Any] | None:
    """The first reveal sentence starting within `window` seconds after `after`."""
    for sentence in sentences:
        start = float(sentence["start"])
        if start < after - 1e-6:
            continue
        if start - after > window:
            break
        if is_reveal(_text(sentence)):
            return sentence
    return None


def complete_reveal(
    span: tuple[float, float],
    sentences: Sequence[dict[str, Any]],
    *,
    window: float,
    ceiling: float,
) -> tuple[float, float]:
    """Extend a span that stops just before its reveal.

    Gated on a reveal actually following, not on promise language alone:
    "you'd hope it gets better year after year" promises something and is
    followed by nothing of the sort, and must be left where it is.
    """
    start, end = span
    reveal = next_reveal(sentences, end, window=window)
    if reveal is None:
        return span

    target = float(reveal["end"])
    # Carry the sentence after the reveal too when it elaborates on it; a bare
    # "The first folding iPhone." lands better with the line that follows.
    following = next(
        (s for s in sentences if float(s["start"]) >= target - 1e-6), None
    )
    if following is not None and float(following["end"]) - start <= ceiling:
        target = float(following["end"])

    if target - start > ceiling:
        return span
    return (round(start, 3), round(max(end, target), 3))
