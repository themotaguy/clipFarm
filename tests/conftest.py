"""Shared fixtures. Nothing here touches Whisper, Ollama or ffmpeg."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def make_segment(idx: int, start: float, text: str, *, word_dur: float = 0.35) -> dict:
    """A Whisper-shaped segment with evenly spaced word timings."""
    tokens = text.split()
    words = [
        {
            "word": tok,
            "start": round(start + i * word_dur, 3),
            "end": round(start + (i + 1) * word_dur, 3),
            "prob": 0.9,
        }
        for i, tok in enumerate(tokens)
    ]
    return {
        "id": idx,
        "start": round(start, 3),
        "end": round(start + len(tokens) * word_dur, 3),
        "text": text,
        "words": words,
        "avg_logprob": -0.25,
    }


@pytest.fixture
def segment_factory():
    return make_segment


@pytest.fixture
def transcript():
    """Twelve segments, ~4.2s each, spanning roughly 50 seconds."""
    sentences = [
        "Here is the thing nobody tells you about inference cost today.",
        "The model is the easy part and serving it is where products die.",
        "You are paying for a sports car and driving it in a parking lot.",
        "Continuous batching gives you ten times throughput for almost free.",
        "Prefix caching turns two seconds into two hundred milliseconds flat.",
        "The third move is simply not calling the large model at all.",
        "A small classifier decides whether the question needs the big one.",
        "Most teams skip all of this and just buy more hardware instead.",
        "The bill always arrives and usually at the worst possible moment.",
        "Measure your cost per request on day one and not on day ninety.",
        "Put it on a dashboard right next to your latency and error rate.",
        "That is the whole game and I will see you all again next time.",
    ]
    segs, t = [], 0.0
    for i, s in enumerate(sentences):
        seg = make_segment(i, t, s)
        segs.append(seg)
        t = seg["end"] + 0.1
    return segs
