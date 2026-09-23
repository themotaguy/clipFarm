"""Deciding where to crop a landscape frame so the subject stays in shot.

Filling a 9:16 canvas from 16:9 footage means throwing away about two thirds of
every frame, so *which* third is kept matters. A centre crop loses a presenter
who sits off to one side; the blurred-letterbox alternative keeps everything but
wastes most of the screen on a smeared copy of the picture.

The subject is located from two cheap signals over a handful of sampled frames:

* **Detail.** A face and body carry far more high-frequency texture than a wall,
  a desk or a bokeh background.
* **Motion.** Between frames, a talking person changes; the set behind them does
  not.

The best crop window is then the one maximising that score, with a mild pull
towards the centre so a busy edge (a window, a caption bar) cannot drag the
frame off a centred speaker.

Face detection would be more precise, but OpenCV 5 ships no cascades and its
DNN detector needs a model downloaded at runtime; this needs nothing but numpy.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Sequence

from app.core import media, timeline

#: Width in pixels that sampled frames are scaled to before analysis.
SAMPLE_WIDTH = 320

#: Strength of the pull towards the middle of the frame, 0..1. Kept light: at
#: 0.35 it dragged a subject at 75% of frame width back to 57%, cropping them
#: out. It only needs to break ties on a flat frame, not overrule a clear peak.
CENTRE_BIAS = 0.12

#: Columns are smoothed over this fraction of the frame width.
SMOOTH_FRACTION = 0.06


def _sample_times(spans: Sequence[timeline.Span], count: int) -> list[float]:
    """Evenly spaced source timestamps across the spans, skipping the edges."""
    total = timeline.total_duration(spans)
    if total <= 0 or count <= 0:
        return []
    times: list[float] = []
    for i in range(count):
        # Offset by half a step so we never land exactly on a cut.
        position = (i + 0.5) / count * total
        running = 0.0
        for start, end in spans:
            length = end - start
            if position <= running + length:
                times.append(start + (position - running))
                break
            running += length
    return times


def extract_frames(
    source: str | Path,
    times: Sequence[float],
    out_dir: str | Path,
    *,
    width: int = SAMPLE_WIDTH,
) -> list[Path]:
    """Grab one small JPEG per timestamp. Missing frames are skipped."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames: list[Path] = []
    for i, t in enumerate(times):
        dest = out_dir / f"frame-{i:03d}.jpg"
        try:
            subprocess.run(
                [media.binary("ffmpeg"), "-y", "-hide_banner", "-loglevel", "error",
                 "-ss", f"{max(t, 0):.3f}", "-i", str(source),
                 "-frames:v", "1", "-vf", f"scale={width}:-2", str(dest)],
                capture_output=True, timeout=30, check=False,
            )
        except (subprocess.SubprocessError, OSError, media.MediaError):
            continue
        if dest.exists() and dest.stat().st_size > 0:
            frames.append(dest)
    return frames


def _smooth(values: Any, window: int) -> Any:
    import numpy as np

    if window <= 1:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def column_scores(frames: Sequence[str | Path]) -> Any | None:
    """Per-column interest across the sampled frames, normalised to 0..1."""
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return None

    greys: list[Any] = []
    for path in frames:
        try:
            with Image.open(path) as img:
                greys.append(np.asarray(img.convert("L"), dtype=np.float32))
        except Exception:      # noqa: BLE001 - a corrupt sample must not be fatal
            continue
    if not greys:
        return None

    width = min(g.shape[1] for g in greys)
    greys = [g[:, :width] for g in greys]

    # Detail: how much the image changes vertically within each column.
    detail = np.zeros(width, dtype=np.float32)
    for g in greys:
        detail += np.abs(np.diff(g, axis=0)).sum(axis=0)

    # Motion: how much each column changes between consecutive samples.
    motion = np.zeros(width, dtype=np.float32)
    for a, b in zip(greys, greys[1:]):
        motion += np.abs(b - a).sum(axis=0)

    def unit(values: Any) -> Any:
        span = float(values.max() - values.min())
        if span <= 0:
            return np.zeros_like(values)
        return (values - values.min()) / span

    score = 0.5 * unit(detail) + 0.5 * unit(motion)
    return _smooth(score, max(3, int(width * SMOOTH_FRACTION)))


def best_window(scores: Any, window_fraction: float) -> float:
    """Centre (0..1) of the highest-scoring crop window.

    A mild centre bias keeps a busy edge from pulling the frame off a subject
    that is, as usual, roughly in the middle.
    """
    import numpy as np

    width = len(scores)
    window = max(1, int(round(width * window_fraction)))
    if window >= width:
        return 0.5

    cumulative = np.concatenate([[0.0], np.cumsum(scores)])
    totals = cumulative[window:] - cumulative[:-window]

    positions = np.arange(len(totals))
    centres = (positions + window / 2) / width
    # Penalise distance from the middle, scaled to the score range.
    spread = float(totals.max() - totals.min()) or 1.0
    penalty = CENTRE_BIAS * spread * np.abs(centres - 0.5) * 2.0

    best = int(np.argmax(totals - penalty))
    return float((best + window / 2) / width)


def window_fraction(
    src_width: int, src_height: int, out_width: int, out_height: int
) -> float:
    """Crop window width as a fraction of the frame width.

    Not the output aspect ratio: cropping 16:9 down to 9:16 keeps only ~32% of
    the width, and searching with a window 1.8x too wide flattens the result.
    """
    if src_width <= 0 or src_height <= 0 or out_height <= 0:
        return 1.0
    crop_w = src_height * (out_width / out_height)
    return max(0.05, min(crop_w / src_width, 1.0))


def find_subject_x(
    source: str | Path,
    spans: Sequence[timeline.Span],
    *,
    window: float,
    samples: int,
    work_dir: str | Path,
) -> float | None:
    """Horizontal centre (0..1) to crop around, or None if it can't be found.

    `window` is the crop width as a fraction of the frame width — use
    `window_fraction()` to derive it from the source and output dimensions.
    """
    times = _sample_times(timeline.normalize(spans), samples)
    if not times:
        return None

    frame_dir = Path(work_dir) / "_framing"
    try:
        frames = extract_frames(source, times, frame_dir)
        if len(frames) < 2:
            return None
        scores = column_scores(frames)
        if scores is None or not len(scores):
            return None
        return best_window(scores, window)
    finally:
        shutil.rmtree(frame_dir, ignore_errors=True)


#: Scene-change score above which two consecutive frames are a cut. 0.35 was
#: measured on real footage: it found all four cuts in a 95s talking-head clip
#: with b-roll inserts and no false positives on camera moves.
SCENE_THRESHOLD = 0.35

#: A "shot" shorter than this is folded into its neighbour. Re-framing for a
#: flash of b-roll is more distracting than leaving the crop where it was.
MIN_SHOT_SECONDS = 1.5

#: Shots whose subject sits within this fraction of the frame width of the
#: previous one keep the previous crop, so the frame does not twitch between
#: two near-identical positions.
MIN_SHIFT = 0.02

#: Frames to analyse per shot regardless of how short it is. Splitting the
#: clip's budget strictly by duration left short shots on three frames, and at
#: that point the estimate is noise — two shots framed 12% apart came back
#: identical and were wrongly merged.
MIN_SHOT_SAMPLES = 6


def find_shots(
    source: str | Path,
    span: timeline.Span,
    *,
    threshold: float = SCENE_THRESHOLD,
) -> list[float]:
    """Cut points inside `span`, in seconds from the span's start.

    One static crop across a whole clip is wrong as soon as the edit cuts: the
    presenter moves, or the shot changes to b-roll framed completely
    differently. Detection runs on downscaled frames — a cut is a cut at 320px.
    """
    start, end = float(span[0]), float(span[1])
    if end - start <= MIN_SHOT_SECONDS:
        return []
    try:
        proc = subprocess.run(
            [media.binary("ffmpeg"), "-y", "-hide_banner", "-nostats",
             "-ss", f"{max(start, 0):.3f}", "-t", f"{end - start:.3f}",
             "-i", str(source), "-an",
             "-filter:v", f"scale={SAMPLE_WIDTH}:-2,"
                          f"select='gt(scene,{threshold})',showinfo",
             "-f", "null", "-"],
            capture_output=True, timeout=300, check=False, text=True,
        )
    except (subprocess.SubprocessError, OSError, media.MediaError):
        return []

    cuts: list[float] = []
    for match in re.finditer(r"pts_time:([0-9.]+)", proc.stderr or ""):
        try:
            cuts.append(float(match.group(1)))
        except ValueError:
            continue
    return sorted(t for t in cuts if MIN_SHOT_SECONDS < t < (end - start))


def _shot_bounds(length: float, cuts: Sequence[float]) -> list[timeline.Span]:
    """Turn cut points into [start, end) shots, dropping the very short ones."""
    edges = [0.0] + [c for c in cuts if 0.0 < c < length] + [length]
    shots: list[timeline.Span] = []
    for a, b in zip(edges, edges[1:]):
        if b - a < MIN_SHOT_SECONDS and shots:
            shots[-1] = (shots[-1][0], b)       # fold a flash into its neighbour
        elif b - a > 0:
            shots.append((a, b))
    if len(shots) > 1 and shots[0][1] - shots[0][0] < MIN_SHOT_SECONDS:
        shots[1] = (shots[0][0], shots[1][1])
        shots.pop(0)
    return shots


def find_subject_track(
    source: str | Path,
    span: timeline.Span,
    *,
    window: float,
    samples: int,
    work_dir: str | Path,
    threshold: float = SCENE_THRESHOLD,
) -> list[dict[str, Any]]:
    """Where to centre the crop over the course of one span, shot by shot.

    Returns `[{"start": s, "end": e, "x": 0..1}, ...]` with times measured from
    the start of the span, which is what the renderer's `t` is relative to.
    Consecutive shots that want near-identical crops are merged, so the result
    is often a single entry — the same answer `find_subject_x` would give.
    """
    start, end = float(span[0]), float(span[1])
    length = end - start
    if length <= 0:
        return []

    shots = _shot_bounds(length, find_shots(source, span, threshold=threshold))
    if len(shots) <= 1:
        x = find_subject_x(source, [(start, end)], window=window,
                           samples=samples, work_dir=work_dir)
        return [] if x is None else [{"start": 0.0, "end": length, "x": x}]

    track: list[dict[str, Any]] = []
    for a, b in shots:
        # Spend the sample budget in proportion to how long each shot lasts,
        # on top of a floor that keeps a brief shot from being guessed at.
        count = max(MIN_SHOT_SAMPLES, int(round(samples * (b - a) / length)))
        x = find_subject_x(source, [(start + a, start + b)], window=window,
                           samples=count, work_dir=work_dir)
        if x is None:
            x = track[-1]["x"] if track else 0.5
        if track and abs(x - track[-1]["x"]) < MIN_SHIFT:
            track[-1]["end"] = b                # not worth moving the frame
            continue
        track.append({"start": a, "end": b, "x": x})
    return track


def average_x(tracks: Sequence[Sequence[dict[str, Any]]]) -> float | None:
    """Duration-weighted mean crop centre across every shot of every span.

    Used as the clip's single reported position, and as the fallback when the
    renderer needs one static box.
    """
    weight = 0.0
    total = 0.0
    for track in tracks:
        for entry in track:
            length = max(float(entry["end"]) - float(entry["start"]), 0.0)
            weight += length
            total += length * float(entry["x"])
    return None if weight <= 0 else total / weight


def crop_geometry(
    src_width: int,
    src_height: int,
    out_width: int,
    out_height: int,
    centre: float | None,
) -> dict[str, int] | None:
    """Pixel crop box that fills the output aspect, centred on `centre`.

    Returns None when the source is not wider than the target, in which case
    there is nothing to crop horizontally.
    """
    if src_width <= 0 or src_height <= 0 or out_width <= 0 or out_height <= 0:
        return None

    target = out_width / out_height
    source = src_width / src_height
    if source <= target + 1e-6:
        return None

    crop_w = int(round(src_height * target))
    crop_w = max(2, min(crop_w, src_width))
    crop_w -= crop_w % 2                      # keep even for yuv420p
    middle = 0.5 if centre is None else min(max(centre, 0.0), 1.0)
    x = int(round(middle * src_width - crop_w / 2))
    x = max(0, min(x, src_width - crop_w))
    return {"width": crop_w, "height": src_height, "x": x - (x % 2), "y": 0}
