"""Choosing cut points that land in pauses rather than on word edges.

Whisper's word timestamps mark where a word's audio nominally starts and ends,
so cutting exactly on one chops the final consonant and leaves no breath — the
clip sounds severed even when the sentence is grammatically complete. Worse, a
segment boundary is not a sentence boundary: Whisper splits on its own decode
windows, so "…not day 90." can be followed immediately by more of the same
thought.

So instead of trusting the boundary we were handed, we look at the surrounding
speech, find the nearest real pause, and cut inside the silence — preferring
pauses that follow sentence-final punctuation.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

SENTENCE_END = (".", "!", "?", "…")
# Characters that end a clause but not a sentence: a weaker, still-usable stop.
CLAUSE_END = (",", ";", ":", "—", "-")


def flatten_words(segments: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """All words from all segments in time order, skipping untimed ones."""
    words = [
        w
        for seg in segments
        for w in (seg.get("words") or [])
        if w.get("start") is not None and w.get("end") is not None
    ]
    words.sort(key=lambda w: (float(w["start"]), float(w["end"])))
    return words


def _text(word: dict[str, Any]) -> str:
    return (word.get("word") or "").strip()


def ends_sentence(word: dict[str, Any]) -> bool:
    return _text(word).endswith(SENTENCE_END)


def ends_clause(word: dict[str, Any]) -> bool:
    return _text(word).endswith(CLAUSE_END)


class Pause:
    """A candidate cut point: a gap between words, or a sentence boundary.

    `cramped` marks a cut with effectively no silence around it and no sentence
    break to justify it — an audible splice mid-thought.
    """

    __slots__ = ("start", "end", "after_sentence", "after_clause", "cramped")

    def __init__(self, start: float, end: float, after_sentence: bool,
                 after_clause: bool, cramped: bool = False) -> None:
        self.start = start
        self.end = end
        self.after_sentence = after_sentence
        self.after_clause = after_clause
        self.cramped = cramped

    @property
    def length(self) -> float:
        return self.end - self.start

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"Pause({self.start:.2f}→{self.end:.2f}, "
                f"{'sentence' if self.after_sentence else 'mid'})")


def find_pauses(
    words: Sequence[dict[str, Any]],
    min_pause: float,
) -> list[Pause]:
    """Gaps *between* consecutive words that are at least `min_pause` long.

    Only real inter-word silence counts. Deliberately nothing is synthesized
    for the run-up to the first word or the run-out after the last: `words` may
    be a fragment of the transcript, in which case inventing a pause from t=0
    would fabricate silence that isn't there. The media edges are handled by
    the caller's floor/ceiling instead.
    """
    pauses: list[Pause] = []
    for prev, nxt in zip(words, words[1:]):
        gap_start, gap_end = float(prev["end"]), float(nxt["start"])
        sentence = ends_sentence(prev)
        # Speakers often run straight on through a full stop, so Whisper leaves
        # almost no gap there. A sentence boundary is still the best place to
        # cut, so it qualifies on punctuation alone.
        if gap_end - gap_start >= min_pause or sentence:
            pauses.append(Pause(
                gap_start, gap_end,
                after_sentence=sentence,
                after_clause=ends_clause(prev) or sentence,
            ))
    return pauses


def _next_word_start(words: Sequence[dict[str, Any]], t: float) -> float | None:
    """Start of the first word beginning at or after `t`."""
    starts = [float(w["start"]) for w in words if float(w["start"]) >= t]
    return min(starts) if starts else None


def _previous_word_end(words: Sequence[dict[str, Any]], t: float) -> float | None:
    """End of the last word finishing at or before `t`."""
    ends = [float(w["end"]) for w in words if float(w["end"]) <= t]
    return max(ends) if ends else None


def _straddling_word(
    words: Sequence[dict[str, Any]], t: float
) -> dict[str, Any] | None:
    """The word being spoken at `t`, if any — cutting here splits a syllable."""
    for w in words:
        if float(w["start"]) < t < float(w["end"]):
            return w
    return None


def align_forward(words: Sequence[dict[str, Any]], t: float) -> float:
    """Earliest word boundary at or after `t`.

    For moving a cut later without landing inside a word — e.g. when a budget
    forbids moving earlier.
    """
    word = _straddling_word(words, t)
    return float(word["end"]) if word is not None else t


def align_back(words: Sequence[dict[str, Any]], t: float) -> float:
    """Latest word boundary at or before `t`."""
    word = _straddling_word(words, t)
    return float(word["start"]) if word is not None else t


def _stay_pause(
    words: Sequence[dict[str, Any]], target: float, min_gap: float = 0.18
) -> Pause:
    """The gap `target` already sits in, as a candidate for not moving at all.

    Without this, a boundary that is *already* a clean sentence end has nothing
    to defend it: if the only nearby pauses lie behind it, the best of a bad set
    wins and the clip gets truncated. Staying put has to be on the ballot.
    """
    # Landing inside a word means there is no silence here whatsoever. Reading
    # the surrounding word ends as a "gap" would score this as a comfortable
    # pause when it is the worst possible cut.
    if _straddling_word(words, target) is not None:
        return Pause(target, target, False, False, cramped=True)

    prev_end = _previous_word_end(words, target)
    next_start = _next_word_start(words, target)
    start = prev_end if prev_end is not None else target
    end = next_start if next_start is not None else target

    after_sentence = after_clause = False
    if prev_end is not None:
        # The word that finishes at this boundary decides how clean it is.
        finishing = [w for w in words if float(w["end"]) == prev_end]
        after_sentence = any(ends_sentence(w) for w in finishing)
        after_clause = after_sentence or any(ends_clause(w) for w in finishing)
    end = max(end, start)
    return Pause(start, end, after_sentence, after_clause,
                 cramped=not after_sentence and (end - start) < min_gap)


#: Cost per second of moving a cut in the direction that *drops* speech the
#: model chose to include, versus the direction that merely adds a little more.
#: Extending is cheap on purpose: reaching a few seconds ahead to let a sentence
#: finish is what a human editor does, and the clip length cap still applies.
TRUNCATE_COST = 2.2
EXTEND_COST = 0.5

#: Charged for cutting where there is neither silence nor a sentence break.
CRAMPED_PENALTY = 3.0


def _score(pause: Pause, target: float, at_start: bool) -> float:
    """Rank a candidate pause: sentence breaks first, then long, then close.

    Distance is deliberately asymmetric. Reaching forward a second to end on a
    finished sentence is nearly free; cutting a second early throws away speech
    the model picked on purpose, so it costs much more.
    """
    score = 0.0
    if pause.after_sentence:
        score += 6.0
    elif pause.after_clause:
        score += 1.5
    # Longer silences are safer cuts, but past ~0.8s there is no extra benefit.
    score += 2.0 * min(pause.length, 0.8)
    if pause.cramped:
        score -= CRAMPED_PENALTY

    anchor = pause.end if at_start else pause.start
    delta = anchor - target
    # At the out-point, earlier truncates; at the in-point, later truncates.
    truncating = delta < 0 if not at_start else delta > 0
    score -= (TRUNCATE_COST if truncating else EXTEND_COST) * abs(delta)
    return score


def _window(
    max_shift: float | None,
    max_extend: float | None,
    max_truncate: float | None,
) -> tuple[float, float]:
    """Resolve how far a cut may travel in each direction.

    `max_shift` sets both at once; the split form lets a cut reach much further
    forward (adding a moment) than backward (discarding one).
    """
    extend = max_extend if max_extend is not None else max_shift
    truncate = max_truncate if max_truncate is not None else max_shift
    return float(extend or 0.0), float(truncate or 0.0)


def snap_end(
    target: float,
    words: Sequence[dict[str, Any]],
    *,
    min_pause: float,
    tail: float,
    ceiling: float,
    max_shift: float | None = None,
    max_extend: float | None = None,
    max_truncate: float | None = None,
) -> float:
    """Pick an out-point at or near `target` that lands in silence.

    Returns a time that sits inside a pause when one is close enough, otherwise
    `target` plus a short tail so the last word is not clipped mid-decay.
    """
    ceiling = max(0.0, ceiling)
    extend, truncate = _window(max_shift, max_extend, max_truncate)
    stay = _stay_pause(words, target, min_pause)
    pauses = [
        p for p in find_pauses(words, min_pause)
        if -truncate <= p.start - target <= extend
    ]
    best = max([*pauses, stay], key=lambda p: _score(p, target, at_start=False))

    if best is not stay:
        # Let the word finish, then cut within the silence rather than at its edge.
        return min(best.start + min(tail, best.length * 0.6), ceiling)

    # Holding position. If the target is mid-word, move to that word's edge —
    # its end when there is room, otherwise back to where it began.
    straddling = _straddling_word(words, target)
    if straddling is not None:
        word_end, word_start = float(straddling["end"]), float(straddling["start"])
        return word_end if word_end <= ceiling else max(word_start, 0.0)

    # Otherwise pad towards the next word but never into it, or we would trade
    # a clipped decay for an audible fragment of the following word.
    limit = _next_word_start(words, target)
    return min(target + tail, limit if limit is not None else ceiling, ceiling)


def snap_start(
    target: float,
    words: Sequence[dict[str, Any]],
    *,
    min_pause: float,
    lead_in: float,
    floor: float = 0.0,
    max_shift: float | None = None,
    max_extend: float | None = None,
    max_truncate: float | None = None,
) -> float:
    """Pick an in-point at or near `target` that starts just after a pause."""
    floor = max(0.0, floor)
    extend, truncate = _window(max_shift, max_extend, max_truncate)
    stay = _stay_pause(words, target, min_pause)
    pauses = [
        p for p in find_pauses(words, min_pause)
        if -extend <= p.end - target <= truncate
    ]
    best = max([*pauses, stay], key=lambda p: _score(p, target, at_start=True))

    if best is not stay:
        # Begin slightly inside the silence so the first word has a run-up.
        return max(best.end - min(lead_in, best.length * 0.6), floor)

    # Holding position. A target inside a word begins on a half-syllable, so
    # start at that word instead.
    straddling = _straddling_word(words, target)
    if straddling is not None:
        return max(float(straddling["start"]), floor)

    # Otherwise back off towards the previous word but never into it.
    limit = _previous_word_end(words, target)
    return max(target - lead_in, limit if limit is not None else floor, floor)


def snap_span(
    start: float,
    end: float,
    words: Sequence[dict[str, Any]],
    *,
    min_pause: float,
    lead_in: float,
    tail: float,
    max_shift: float | None = None,
    max_extend: float | None = None,
    max_truncate: float | None = None,
    media_duration: float | None = None,
    min_seconds: float | None = None,
    max_seconds: float | None = None,
) -> tuple[float, float]:
    """Snap both edges of a span, keeping it ordered and inside its limits."""
    ceiling = media_duration if media_duration else end + tail
    extend, truncate = _window(max_shift, max_extend, max_truncate)
    window = {"max_shift": max_shift, "max_extend": max_extend,
              "max_truncate": max_truncate}
    new_start = snap_start(
        start, words, min_pause=min_pause, lead_in=lead_in, **window
    )
    new_end = snap_end(
        end, words, min_pause=min_pause, tail=tail, ceiling=ceiling, **window
    )

    if new_end - new_start < 0.5:          # snapping collapsed the span
        new_start, new_end = start, max(end, start + 0.5)

    # Snapping can push a span past its length limits. Clamping straight to
    # `new_start + limit` would drop the cut at an arbitrary instant, quite
    # possibly mid-word, undoing the work above — so re-snap to a real
    # boundary at or under the cap instead.
    if max_seconds and new_end - new_start > max_seconds:
        limit = new_start + max_seconds
        new_end = snap_end(
            limit, words, min_pause=min_pause, tail=0.0, ceiling=limit,
            max_extend=0.0, max_truncate=max(truncate, 3.0),
        )
    if min_seconds and new_end - new_start < min_seconds:
        wanted = min(new_start + min_seconds, ceiling)
        new_end = snap_end(
            wanted, words, min_pause=min_pause, tail=tail, ceiling=ceiling,
            max_extend=max(extend, 3.0), max_truncate=0.0,
        )

    return round(new_start, 3), round(new_end, 3)
