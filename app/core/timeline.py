"""Mapping between source time and clip time when a clip is stitched.

A clip used to be one continuous slice, so clip time was just
`source_time - start`. Once a clip can prepend an earlier "setup" span, the
output timeline is a concatenation: several disjoint source ranges laid end to
end. Captions, SRT sidecars and the render filtergraph all need the same
mapping, so it lives here once.

A span is a plain `(start, end)` pair in source-media seconds.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

Span = tuple[float, float]


def normalize(spans: Iterable[Any]) -> list[Span]:
    """Coerce assorted span shapes into ordered, non-empty `(start, end)` pairs.

    Accepts pairs, dicts with `start`/`end` keys, or a flat `(start, end)`.
    Spans are sorted and any that are empty or inverted are dropped.
    """
    out: list[Span] = []
    for item in spans or ():
        if isinstance(item, dict):
            start, end = float(item["start"]), float(item["end"])
        else:
            start, end = float(item[0]), float(item[1])
        if end > start:
            out.append((start, end))
    out.sort()
    return out


def as_spans(value: Any) -> list[Span]:
    """Normalize either a single `(start, end)` pair or a list of spans.

    Callers hand over both shapes — a plain clip is one pair, a stitched clip
    is a list — so accept either rather than making every call site wrap.
    """
    if value is None:
        return []
    if (
        isinstance(value, (tuple, list))
        and len(value) == 2
        and all(isinstance(v, (int, float)) for v in value)
    ):
        return normalize([value])
    return normalize(value)


def total_duration(spans: Sequence[Span]) -> float:
    """Length of the rendered clip: the sum of its spans, not their envelope."""
    return sum(e - s for s, e in spans)


def offsets(spans: Sequence[Span]) -> list[float]:
    """Clip-time position at which each span begins."""
    out: list[float] = []
    running = 0.0
    for start, end in spans:
        out.append(running)
        running += end - start
    return out


def to_clip_time(spans: Sequence[Span], t: float) -> float | None:
    """Convert a source time to clip time, or None if no span contains it."""
    for (start, end), offset in zip(spans, offsets(spans)):
        if start <= t <= end:
            return offset + (t - start)
    return None


def clamp_to_clip(spans: Sequence[Span], t: float) -> float:
    """Convert a source time to clip time, snapping to the nearest span edge."""
    mapped = to_clip_time(spans, t)
    if mapped is not None:
        return mapped
    # Outside every span: attach it to whichever edge is closest.
    best: tuple[float, float] | None = None
    for (start, end), offset in zip(spans, offsets(spans)):
        for edge, mapped_edge in ((start, offset), (end, offset + (end - start))):
            distance = abs(t - edge)
            if best is None or distance < best[0]:
                best = (distance, mapped_edge)
    return best[1] if best else 0.0


def words_in_spans(
    segments: Iterable[dict[str, Any]],
    spans: Sequence[Span],
) -> list[dict[str, Any]]:
    """Words falling inside `spans`, with times rebased onto the clip timeline.

    A word straddling a cut is clipped to the span rather than dropped, so the
    caption still shows it for as long as it is audible.
    """
    words: list[dict[str, Any]] = []
    for (span_start, span_end), offset in zip(spans, offsets(spans)):
        for seg in segments:
            if seg["end"] <= span_start or seg["start"] >= span_end:
                continue
            for w in seg.get("words") or []:
                w_start, w_end = float(w["start"]), float(w["end"])
                if w_end <= span_start or w_start >= span_end:
                    continue
                token = (w.get("word") or "").strip()
                if not token:
                    continue
                lo = max(w_start, span_start)
                hi = min(w_end, span_end)
                words.append({
                    "word": token,
                    "start": round(offset + (lo - span_start), 3),
                    "end": round(offset + (hi - span_start), 3),
                })
    words.sort(key=lambda w: (w["start"], w["end"]))
    return words


def segments_in_spans(
    segments: Iterable[dict[str, Any]],
    spans: Sequence[Span],
) -> list[dict[str, Any]]:
    """Transcript segments overlapping `spans`, rebased onto the clip timeline."""
    out: list[dict[str, Any]] = []
    for (span_start, span_end), offset in zip(spans, offsets(spans)):
        for seg in segments:
            if seg["end"] <= span_start or seg["start"] >= span_end:
                continue
            lo = max(float(seg["start"]), span_start)
            hi = min(float(seg["end"]), span_end)
            if hi <= lo:
                continue
            out.append({
                **seg,
                "start": round(offset + (lo - span_start), 3),
                "end": round(offset + (hi - span_start), 3),
            })
    out.sort(key=lambda s: (s["start"], s["end"]))
    return out


def merge_touching(spans: Sequence[Span], gap: float = 0.05) -> list[Span]:
    """Fuse spans that abut or overlap, so a 'stitch' of one slice stays one cut."""
    merged: list[Span] = []
    for start, end in normalize(spans):
        if merged and start - merged[-1][1] <= gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged
