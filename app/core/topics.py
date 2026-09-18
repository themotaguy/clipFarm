"""Finding where a topic actually finishes, not just where a sentence does.

Snapping a cut to a full stop stops clips sounding severed, but a grammatical
sentence is not a complete thought: "So there are three moves." is a fine
sentence and a terrible place to end, because the clip then promises something
it never delivers.

Two signals locate the real end of a thought:

* **Discourse markers.** Speakers announce turns explicitly — "So,", "Now,",
  "Here's the thing", "Let me give you a concrete number". A marker opening a
  sentence means the previous thought closed just before it.
* **Breathing.** People pause noticeably longer between points than between
  sentences of the same point. On real footage the median gap at a sentence end
  is ~0.26s, while a topic change runs to half a second or more.
* **Semantic drift.** Embeddings of neighbouring passages diverge when the
  subject changes. This is TextTiling: score each sentence gap by how much the
  block before it resembles the block after, and treat the local minima as
  boundaries.

Markers and pauses work offline and cost nothing, so they are always used. The
embedding pass is optional and simply sharpens the result — on a continuous
monologue it adds very little, because everything is genuinely on topic.
"""

from __future__ import annotations

import math
import re
from typing import Any, Callable, Iterable, Sequence

from app.core import context

EmbedFn = Callable[[Sequence[str]], Sequence[Sequence[float]]]

#: Sentence openings that announce a new thought, so the one before it ended.
TOPIC_MARKERS = (
    "so ", "so,", "now ", "now,", "okay", "ok,", "alright", "all right",
    "anyway", "anyhow", "moving on", "next ", "next,", "let me", "let's",
    "here is", "here's", "but here", "and here", "the point is",
    "the thing is", "my point", "look,", "listen,", "first,", "second,",
    "third,", "finally", "lastly", "in short", "to be clear", "bottom line",
    "the question is", "which brings", "that said", "meanwhile",
    "on the other hand", "by the way", "speaking of",
)

#: Openings that wrap something up — a good place to end *after*.
CLOSING_MARKERS = (
    "so that is", "so that's", "that is why", "that's why", "that is the",
    "that's the", "in other words", "which is why", "the lesson",
    "the takeaway", "bottom line", "and that is", "and that's",
    "thanks for", "thank you for", "see you", "i will see you",
)

#: How many sentences either side of a gap the semantic comparison looks at.
BLOCK_SENTENCES = 3

#: A sentence end followed by at least this much silence reads as a new point.
TOPIC_PAUSE_SECONDS = 0.45


def _starts_with_marker(text: str, markers: Iterable[str]) -> bool:
    lowered = re.sub(r"^[^a-z0-9]+", "", (text or "").strip().lower())
    return any(lowered.startswith(m) for m in markers)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _mean(vectors: Sequence[Sequence[float]]) -> list[float]:
    if not vectors:
        return []
    width = len(vectors[0])
    return [sum(v[i] for v in vectors) / len(vectors) for i in range(width)]


def marker_boundaries(sentences: Sequence[dict[str, Any]]) -> set[int]:
    """Gap indices (between sentence i and i+1) implied by discourse markers.

    A marker at the start of sentence i+1 closes the thought at the end of
    sentence i, which is gap index i.
    """
    gaps: set[int] = set()
    for i in range(len(sentences) - 1):
        if _starts_with_marker(sentences[i + 1]["text"], TOPIC_MARKERS):
            gaps.add(i)
        # A closing sentence ends a thought where it itself ends.
        if _starts_with_marker(sentences[i]["text"], CLOSING_MARKERS):
            gaps.add(i)
    return gaps


def pause_boundaries(
    sentences: Sequence[dict[str, Any]],
    *,
    min_pause: float = TOPIC_PAUSE_SECONDS,
) -> set[int]:
    """Gap indices where the speaker took a real breath after a sentence."""
    gaps: set[int] = set()
    for i in range(len(sentences) - 1):
        if sentences[i + 1]["start"] - sentences[i]["end"] >= min_pause:
            gaps.add(i)
    return gaps


def semantic_boundaries(
    sentences: Sequence[dict[str, Any]],
    embed: EmbedFn,
    *,
    threshold: float,
) -> set[int]:
    """Gap indices where the passage either side diverges semantically."""
    if len(sentences) < 2 * BLOCK_SENTENCES:
        return set()

    try:
        vectors = list(embed([s["text"] for s in sentences]))
    except Exception:      # noqa: BLE001 - embeddings are an optional refinement
        return set()
    if len(vectors) != len(sentences) or not vectors or not vectors[0]:
        return set()

    scores: list[tuple[int, float]] = []
    for gap in range(len(sentences) - 1):
        before = vectors[max(0, gap - BLOCK_SENTENCES + 1):gap + 1]
        after = vectors[gap + 1:gap + 1 + BLOCK_SENTENCES]
        if not before or not after:
            continue
        scores.append((gap, _cosine(_mean(before), _mean(after))))

    if not scores:
        return set()

    # Keep gaps that are both weakly coherent in absolute terms and a local
    # dip: a run of uniformly low scores is a rambling passage, not a seam.
    boundaries: set[int] = set()
    for idx, (gap, score) in enumerate(scores):
        if score > threshold:
            continue
        prev_score = scores[idx - 1][1] if idx > 0 else 1.0
        next_score = scores[idx + 1][1] if idx + 1 < len(scores) else 1.0
        if score <= prev_score and score <= next_score:
            boundaries.add(gap)
    return boundaries


def find_boundaries(
    words: Sequence[dict[str, Any]],
    *,
    embed: EmbedFn | None = None,
    threshold: float = 0.62,
    min_pause: float = TOPIC_PAUSE_SECONDS,
) -> list[float]:
    """Times (seconds) at which a topic finishes, in order."""
    sentences = context.sentences(words)
    if not sentences:
        return []

    gaps = marker_boundaries(sentences)
    gaps |= pause_boundaries(sentences, min_pause=min_pause)
    if embed is not None:
        gaps |= semantic_boundaries(sentences, embed, threshold=threshold)

    # A gap index i means "the thought ends when sentence i ends".
    times = sorted(float(sentences[i]["end"]) for i in gaps if i < len(sentences))
    # The end of the transcript always closes whatever was in progress.
    last = float(sentences[-1]["end"])
    if not times or times[-1] < last:
        times.append(last)
    return times


def next_boundary(boundaries: Sequence[float], after: float) -> float | None:
    """First topic end at or after `after`."""
    later = [b for b in boundaries if b >= after - 1e-6]
    return min(later) if later else None


def complete_topic(
    span: tuple[float, float],
    boundaries: Sequence[float],
    *,
    max_extend: float,
    max_seconds: float,
) -> tuple[float, float]:
    """Run a span's end on to the next topic boundary, within budget.

    Leaves the span alone when the nearest boundary is further than
    `max_extend` or would push it past `max_seconds` — better a clip that ends
    on a clean sentence than one padded with an unrelated minute.
    """
    start, end = span
    target = next_boundary(boundaries, end)
    if target is None or target <= end:
        return span
    if target - end > max_extend:
        return span
    if target - start > max_seconds:
        return span
    return (start, target)
