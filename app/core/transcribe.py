"""Stage 2 — Whisper transcription with word-level timestamps.

`faster-whisper` returns a *generator* of segments, so we consume it lazily and
hand each segment to a callback. That is what lets the indexing stage start
embedding text while the back half of the video is still being decoded.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable, Iterator

import config

_model = None
_model_key: tuple[str, str, str] | None = None
_model_lock = threading.Lock()

SegmentCallback = Callable[[dict[str, Any]], None]


def get_model():
    """Load (and cache) the Whisper model. Downloads on first use."""
    global _model, _model_key
    key = (config.WHISPER_MODEL, config.WHISPER_DEVICE, config.WHISPER_COMPUTE_TYPE)
    with _model_lock:
        if _model is None or _model_key != key:
            from faster_whisper import WhisperModel

            _model = WhisperModel(
                config.WHISPER_MODEL,
                device=config.WHISPER_DEVICE,
                compute_type=config.WHISPER_COMPUTE_TYPE,
            )
            _model_key = key
    return _model


def transcribe(
    audio_path: str | Path,
    duration: float,
    *,
    on_segment: SegmentCallback | None = None,
    on_progress: Callable[[float, str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Transcribe `audio_path`, streaming each finished segment to `on_segment`.

    Returns `(segments, info)` where each segment carries word-level timings.
    """
    model = get_model()

    segment_iter, info = model.transcribe(
        str(audio_path),
        language=config.WHISPER_LANGUAGE,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 400},
        beam_size=5,
        condition_on_previous_text=False,   # avoids runaway repetition on long media
    )

    segments: list[dict[str, Any]] = []
    total = duration or getattr(info, "duration", 0.0) or 0.0

    for idx, seg in enumerate(_iter_segments(segment_iter)):
        if should_cancel and should_cancel():
            break

        words = [
            {
                "word": w.word,
                "start": round(float(w.start), 3),
                "end": round(float(w.end), 3),
                "prob": round(float(getattr(w, "probability", 0.0) or 0.0), 4),
            }
            for w in (seg.words or [])
            if w.start is not None and w.end is not None
        ]
        record = {
            "id": idx,
            "start": round(float(seg.start), 3),
            "end": round(float(seg.end), 3),
            "text": (seg.text or "").strip(),
            "words": words,
            "avg_logprob": round(float(getattr(seg, "avg_logprob", 0.0) or 0.0), 4),
        }
        if not record["text"]:
            continue

        segments.append(record)
        if on_segment:
            on_segment(record)
        if on_progress and total > 0:
            on_progress(
                min(record["end"] / total, 1.0),
                f"Transcribed {_hms(record['end'])} / {_hms(total)}",
            )

    meta = {
        "language": getattr(info, "language", None),
        "language_probability": round(float(getattr(info, "language_probability", 0.0) or 0.0), 4),
        "model": config.WHISPER_MODEL,
        "segment_count": len(segments),
        "word_count": sum(len(s["words"]) for s in segments),
    }
    return segments, meta


def _iter_segments(segment_iter) -> Iterator[Any]:
    """faster-whisper yields lazily; this just makes the intent explicit."""
    for seg in segment_iter:
        yield seg


def _hms(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def full_text(segments: list[dict[str, Any]]) -> str:
    return " ".join(s["text"] for s in segments).strip()


def to_srt(segments: list[dict[str, Any]], offset: float = 0.0) -> str:
    """Render segments as an SRT file body, shifted by `offset` seconds."""
    lines: list[str] = []
    for i, seg in enumerate(segments, start=1):
        start = max(0.0, seg["start"] - offset)
        end = max(0.0, seg["end"] - offset)
        lines.append(str(i))
        lines.append(f"{_srt_time(start)} --> {_srt_time(end)}")
        lines.append(seg["text"])
        lines.append("")
    return "\n".join(lines)


def _srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
