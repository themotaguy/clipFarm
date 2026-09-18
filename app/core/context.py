"""Detecting clips that open on a reference they never explain.

A clip that begins "The third move is simply not calling the model" is fine in
the middle of a talk and baffling on its own: nothing inside it says what the
first two moves were. The fix is to prepend the earlier passage that introduces
the referent, which is what the stitching support in `render` exists for.

Asking the LLM to spot this works poorly with a local 8B model — it reliably
answers that the clip is self-contained and rates its own clarity 9/10. So the
detection here is deterministic: find the referring expression, recover the
noun it points at, and look backwards for the sentence that introduced it.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

SENTENCE_END = (".", "!", "?", "…")

#: Ordinals that imply an earlier enumeration ("the third move" needs moves 1-2).
ORDINALS = (
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh",
    "eighth", "ninth", "tenth", "last", "other", "next", "final", "same",
    "latter", "former",
)

#: Determiners that point outside the clip when followed by a noun.
DEMONSTRATIVES = ("this", "that", "these", "those")

#: Bare pronoun subjects with no antecedent inside the clip.
DANGLING_PRONOUNS = (
    "it", "its", "they", "them", "their", "he", "she", "his", "her",
    "this", "that", "these", "those", "both", "neither", "such",
)

#: Phrases that explicitly point back at earlier material.
BACKREFERENCES = (
    "as i said", "as i mentioned", "like i said", "like i mentioned",
    "as mentioned", "as above", "going back", "that is why", "which is why",
    "so that", "therefore", "for that reason", "the reason is",
)

_WORD_RE = re.compile(r"[a-z0-9']+")

#: Nouns too generic to be worth searching backwards for.
_STOP_NOUNS = {
    "thing", "things", "one", "ones", "way", "ways", "part", "parts",
    "time", "times", "case", "cases", "point", "points", "kind", "sort",
}


def sentences(words: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group word timings into timed sentences."""
    out: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    for word in words:
        current.append(word)
        if (word.get("word") or "").strip().endswith(SENTENCE_END):
            out.append(_sentence(current))
            current = []
    if current:
        out.append(_sentence(current))
    return out


def _sentence(words: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "text": " ".join((w.get("word") or "").strip() for w in words).strip(),
        "start": float(words[0]["start"]),
        "end": float(words[-1]["end"]),
    }


def _tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def dangling_reference(text: str) -> dict[str, Any] | None:
    """Whether `text` opens on something it does not explain.

    Returns `{"kind": ..., "term": ...}` where `term` is the noun to search
    backwards for (may be empty when there is no useful noun), or None when the
    opening stands on its own.
    """
    lowered = (text or "").strip().lower()
    if not lowered:
        return None

    for phrase in BACKREFERENCES:
        if lowered.startswith(phrase):
            return {"kind": "backreference", "term": ""}

    tokens = _tokens(lowered)
    if not tokens:
        return None

    # "the third move" / "the other option" -> recover the head noun.
    window = tokens[:4]
    for i, token in enumerate(window):
        if token in ORDINALS or token in DEMONSTRATIVES:
            noun = next(
                (t for t in tokens[i + 1:i + 3] if t not in _STOP_NOUNS and len(t) > 2),
                "",
            )
            # A demonstrative with no noun after it is a bare pronoun instead.
            if token in DEMONSTRATIVES and not noun and i > 0:
                continue
            return {"kind": "enumeration" if token in ORDINALS else "demonstrative",
                    "term": noun}

    if tokens[0] in DANGLING_PRONOUNS:
        return {"kind": "pronoun", "term": ""}

    return None


def _mentions(text: str, term: str) -> bool:
    """Whether `text` mentions `term`, tolerating a plural s."""
    if not term:
        return False
    tokens = set(_tokens(text))
    variants = {term, f"{term}s", term.rstrip("s")}
    return bool(tokens & variants)


def suggest_setup(
    words: Sequence[dict[str, Any]],
    main_start: float,
    *,
    lookback: float,
    max_seconds: float,
    min_seconds: float = 1.5,
) -> tuple[float, float] | None:
    """Find an earlier passage that supplies what the clip opens by referencing.

    Returns a `(start, end)` span to play before the clip, or None when the
    clip explains itself or nothing suitable is close enough.
    """
    if not words:
        return None

    all_sentences = sentences(words)
    inside = [s for s in all_sentences if s["end"] > main_start]
    before = [
        s for s in all_sentences
        if s["end"] <= main_start + 0.01 and s["start"] >= main_start - lookback
    ]
    if not inside or not before:
        return None

    opening = " ".join(s["text"] for s in inside[:2])
    reference = dangling_reference(opening)
    if not reference:
        return None

    term = reference["term"]
    # Prefer the most recent earlier sentence that actually names the referent;
    # otherwise fall back to the sentence immediately before the clip.
    chosen: list[dict[str, Any]] = []
    if term:
        # Don't count a mention that is already inside the clip.
        matches = [s for s in before if _mentions(s["text"], term)]
        if matches:
            chosen = [matches[-1]]
    if not chosen:
        chosen = [before[-1]]

    start, end = chosen[0]["start"], chosen[-1]["end"]

    # Very short antecedents ("There are three moves.") often need the sentence
    # after them to make sense; extend while there is budget.
    idx = before.index(chosen[-1])
    while end - start < min_seconds + 1.0 and idx + 1 < len(before):
        idx += 1
        end = before[idx]["end"]

    if end - start > max_seconds:
        start = end - max_seconds          # keep the tail, nearest the payoff
    if end - start < min_seconds:
        return None
    return (round(start, 3), round(min(end, main_start), 3))
