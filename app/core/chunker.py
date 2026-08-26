"""Stage 3 — turn a stream of Whisper segments into overlapping, timed windows.

Windows are the unit of both retrieval and clipping, so they are built on the
time axis (target ~45 s with 50% overlap) rather than on raw character counts.
LangChain's recursive splitter is used as a guard: any window whose text runs
long for the embedding model is split at sentence boundaries and each piece is
re-anchored to real word timestamps.
"""

from __future__ import annotations

from typing import Any, Iterator

import config

MAX_CHUNK_CHARS = 2400
CHUNK_OVERLAP_CHARS = 200

_splitter = None


def _get_splitter():
    global _splitter
    if _splitter is None:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        _splitter = RecursiveCharacterTextSplitter(
            chunk_size=MAX_CHUNK_CHARS,
            chunk_overlap=CHUNK_OVERLAP_CHARS,
            separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "],
            keep_separator=True,
        )
    return _splitter


def _words_of(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten segment words, synthesizing one per segment if timings are missing."""
    words: list[dict[str, Any]] = []
    for seg in segments:
        if seg.get("words"):
            words.extend(seg["words"])
        elif seg.get("text"):
            words.append({"word": seg["text"], "start": seg["start"], "end": seg["end"]})
    return words


def _render(words: list[dict[str, Any]]) -> tuple[str, list[tuple[int, int, int]]]:
    """Join word tokens into text, returning (char_start, char_end, word_index) spans.

    Building the text from the same tokens we index means character offsets map
    back to word timestamps exactly, with no fuzzy matching.
    """
    parts: list[str] = []
    spans: list[tuple[int, int, int]] = []
    pos = 0
    for idx, w in enumerate(words):
        token = (w.get("word") or "").strip()
        if not token:
            continue
        if pos > 0:
            parts.append(" ")
            pos += 1
        spans.append((pos, pos + len(token), idx))
        parts.append(token)
        pos += len(token)
    return "".join(parts), spans


class WindowBuilder:
    """Incremental sliding-window builder. Feed it segments, take windows out."""

    def __init__(
        self,
        target: float | None = None,
        min_seconds: float | None = None,
        max_seconds: float | None = None,
        overlap: float = 0.5,
    ) -> None:
        self.target = target or config.CLIP_TARGET_SECONDS
        self.min_seconds = min_seconds or config.CLIP_MIN_SECONDS
        self.max_seconds = max_seconds or config.CLIP_MAX_SECONDS
        self.overlap = min(max(overlap, 0.0), 0.9)
        self._buf: list[dict[str, Any]] = []
        self._emitted = 0
        self._last_span: tuple[float, float] | None = None

    def push(self, segment: dict[str, Any]) -> list[dict[str, Any]]:
        """Add one transcript segment; return any windows it completed."""
        self._buf.append(segment)
        out: list[dict[str, Any]] = []
        while self._buf and self._span(self._buf) >= self.target:
            out.extend(self._emit(self._buf))
            self._advance()
        return out

    def finish(self) -> list[dict[str, Any]]:
        """Flush the tail of the transcript."""
        out: list[dict[str, Any]] = []
        while self._buf:
            if self._span(self._buf) < self.min_seconds and self._emitted:
                break                       # trailing scrap, already covered by overlap
            out.extend(self._emit(self._buf))
            before = len(self._buf)
            self._advance()
            if len(self._buf) >= before:    # defensive: guarantee forward progress
                break
        self._buf = []
        return out

    # --- internals ---

    @staticmethod
    def _span(segs: list[dict[str, Any]]) -> float:
        return segs[-1]["end"] - segs[0]["start"]

    def _advance(self) -> None:
        """Slide the buffer forward, retaining `overlap` of the window's duration."""
        if not self._buf:
            return
        cut_time = self._buf[0]["start"] + self.target * (1.0 - self.overlap)
        k = 0
        while k < len(self._buf) - 1 and self._buf[k]["end"] <= cut_time:
            k += 1
        self._buf = self._buf[max(k, 1):]

    def _emit(self, segs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # Respect the hard ceiling even if one segment is unusually long.
        start = segs[0]["start"]
        capped: list[dict[str, Any]] = []
        for seg in segs:
            if capped and (seg["end"] - start) > self.max_seconds:
                break
            capped.append(seg)
        if not capped:
            return []

        span = capped[-1]["end"] - capped[0]["start"]
        if span < self.min_seconds and self._emitted:
            return []

        # Skip windows that are ~the same slice of video as the previous one.
        current = (capped[0]["start"], capped[-1]["end"])
        if self._last_span and _iou(current, self._last_span) > 0.9:
            return []

        windows = list(self._build(capped))
        if windows:
            self._last_span = current
            self._emitted += len(windows)
        return windows

    def _build(self, segs: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
        words = _words_of(segs)
        text, spans = _render(words)
        if not text.strip() or not spans:
            return

        if len(text) <= MAX_CHUNK_CHARS:
            yield self._window(text, words, spans[0][2], spans[-1][2], segs)
            return

        # Oversized: split on sentence boundaries, re-anchor each piece to words.
        cursor = 0
        for piece in _get_splitter().split_text(text):
            piece = piece.strip()
            if not piece:
                continue
            at = text.find(piece, cursor)
            if at < 0:
                at = text.find(piece)
            if at < 0:
                continue
            lo, hi = at, at + len(piece)
            cursor = max(cursor, at + max(len(piece) - CHUNK_OVERLAP_CHARS, 1))
            covered = [wi for (cs, ce, wi) in spans if cs < hi and ce > lo]
            if not covered:
                continue
            yield self._window(piece, words, covered[0], covered[-1], segs)

    def _window(
        self,
        text: str,
        words: list[dict[str, Any]],
        first_word: int,
        last_word: int,
        segs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        w_start = float(words[first_word]["start"])
        w_end = float(words[last_word]["end"])
        index = self._emitted
        return {
            "id": f"chunk-{index:05d}",
            "index": index,
            "text": text,
            "start": round(w_start, 3),
            "end": round(w_end, 3),
            "duration": round(w_end - w_start, 3),
            "n_words": last_word - first_word + 1,
            "seg_ids": [s["id"] for s in segs],
        }


def _iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Intersection-over-union of two time spans."""
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def overlap_ratio(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Fraction of the *shorter* span that the two spans share."""
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    shortest = min(a[1] - a[0], b[1] - b[0])
    return inter / shortest if shortest > 0 else 0.0
