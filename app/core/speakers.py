"""Deciding which face on screen is the one talking.

Per-shot framing (see `framing.py`) handles the common case, because an editor
cutting between two people has already done the switching for us. It cannot
help with a locked-off two-shot — a podcast wide, an interview where both
people stay in frame for minutes at a time. There, the crop has to follow the
talker within a single shot, and detail-and-motion scoring cannot tell one
seated person from another: both faces carry texture, and the listener nodding
along out-scores the speaker sitting still.

So this module answers a narrower question with better evidence:

* **Where are the faces?** YuNet, via OpenCV, over frames sampled a few times
  a second, run in a subprocess (see `facedetect.py` for why).
* **Which mouth is moving while words are being said?** We already have
  word-level timings from Whisper, which is a far cleaner speech signal than
  audio energy — it ignores music, room tone and the other person's laughter.
  A face whose mouth region churns during words and settles between them is
  the one speaking.

Everything degrades to `None`, and therefore back to per-shot framing, if the
model cannot be fetched, fewer than two faces are present, or the evidence is
weak. Guessing wrong here is worse than not guessing: a crop that jumps to the
wrong person mid-sentence is far more jarring than one that sits still.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Sequence

import config
from app.core import media, timeline

#: YuNet: ~230 KB, runs on OpenCV's own DNN backend, and returns five
#: landmarks per face including both mouth corners — which is exactly the
#: region we need to watch.
MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
MODEL_NAME = "face_detection_yunet_2023mar.onnx"

#: Frames per second to analyse. Mouths open and close several times a second;
#: below about 4 fps the motion signal washes out.
SAMPLE_FPS = 5.0

#: Faces need pixels to be found. At the 320px used for crop scoring a face in
#: a wide shot is only ~30px across, near the limit of what the detector sees.
SAMPLE_WIDTH = 640

#: Minimum confidence for a detection to count.
MIN_CONFIDENCE = 0.5

#: Two detections whose centres sit within this fraction of the frame width of
#: each other are taken to be the same person.
SAME_FACE_X = 0.08

#: A person must appear in at least this fraction of sampled frames to be a
#: participant rather than someone passing through the background.
MIN_PRESENCE = 0.3

#: Half-size of the mouth patch, as a multiple of the distance between the two
#: mouth corners. 1.0 takes in the whole mouth plus a little jaw, which moves
#: when someone speaks and does not when they only smile.
MOUTH_SCALE = 1.0

#: Never switch the crop for a burst shorter than this. Real conversational
#: turns are seconds long; sub-second flips are detector noise and read as a
#: glitch rather than an edit.
MIN_HOLD_SECONDS = 1.2

#: How much more a face's mouth must move than the runner-up before we believe
#: it is the speaker, as a fraction of the runner-up's score.
MIN_MARGIN = 0.25

#: How often to reconsider who is talking. Speech in a real conversation runs
#: continuously across a change of speaker, so waiting for a gap never
#: reconsiders at all — on a 12s two-hander it produced exactly one window.
WINDOW_SECONDS = 2.0

#: A window needs this many sampled frames of actual speech to be judged.
MIN_SPEECH_FRAMES = 3

#: Shots shorter than this are not worth decoding at 5 fps to interrogate. A
#: brief insert is handled by per-shot framing already.
MIN_SHOT_SECONDS = 4.0


def model_file(cache_dir: str | Path | None = None) -> Path | None:
    """Path to the face model, fetching it once if it is not cached yet.

    Returns None rather than raising: no model simply means no speaker
    tracking, and the caller falls back to per-shot framing.
    """
    directory = Path(cache_dir or config.MODEL_DIR)
    destination = directory / MODEL_NAME
    if destination.exists() and destination.stat().st_size > 0:
        return destination

    directory.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".part")
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=30) as response:
            partial.write_bytes(response.read())
        if partial.stat().st_size == 0:
            raise OSError("empty download")
        partial.replace(destination)
    except (urllib.error.URLError, OSError, ValueError):
        partial.unlink(missing_ok=True)
        return None
    return destination


def available() -> bool:
    """Is face detection usable at all in this environment?"""
    return bool(_worker_ok()) and model_file() is not None


def _worker_ok() -> bool:
    """Can the detection subprocess import OpenCV?"""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", "import cv2"],
            capture_output=True, timeout=60, check=False,
        )
    except (subprocess.SubprocessError, OSError):
        return False
    return proc.returncode == 0


def extract_sequence(
    source: str | Path,
    span: timeline.Span,
    out_dir: str | Path,
    *,
    fps: float = SAMPLE_FPS,
    width: int = SAMPLE_WIDTH,
) -> list[tuple[float, Path]]:
    """Decode the span once into evenly spaced frames.

    One ffmpeg pass, not one per frame: at 5 fps a 95-second clip is nearly
    500 frames, and spawning a process each would cost longer than the render.
    Returns `(seconds from span start, path)` pairs.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    start, end = float(span[0]), float(span[1])
    if end <= start:
        return []

    try:
        subprocess.run(
            [media.binary("ffmpeg"), "-y", "-hide_banner", "-loglevel", "error",
             "-ss", f"{max(start, 0):.3f}", "-t", f"{end - start:.3f}",
             "-i", str(source), "-an",
             "-vf", f"fps={fps},scale={width}:-2",
             str(out_dir / "f-%05d.jpg")],
            capture_output=True, timeout=600, check=False,
        )
    except (subprocess.SubprocessError, OSError, media.MediaError):
        return []

    frames = sorted(out_dir.glob("f-*.jpg"))
    return [(i / fps, path) for i, path in enumerate(frames)]


def detect_faces(
    frames: Sequence[tuple[float, Path]],
    model: str | Path,
) -> list[tuple[float, list[dict[str, Any]]]]:
    """Face boxes per frame, normalised to 0..1, via the isolated worker.

    Detection runs in a subprocess so that neither OpenCV's bundled FFmpeg nor
    a native crash can touch the process doing the render.
    """
    if not frames:
        return []
    frame_dir = frames[0][1].parent
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "app.core.facedetect", str(model), str(frame_dir)],
            capture_output=True, timeout=900, check=False, text=True,
            cwd=str(Path(__file__).resolve().parent.parent.parent),
        )
    except (subprocess.SubprocessError, OSError):
        return []
    if proc.returncode != 0:
        return []

    try:
        payload = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        return []
    if payload.get("error"):
        return []

    by_name = {entry["name"]: entry["boxes"] for entry in payload.get("frames", [])}
    results: list[tuple[float, list[dict[str, Any]]]] = []
    for when, path in frames:
        boxes = [b for b in by_name.get(path.name, [])
                 if float(b.get("score", 0.0)) >= MIN_CONFIDENCE]
        results.append((when, boxes))
    return results


def group_people(
    detections: Sequence[tuple[float, list[dict[str, float]]]],
) -> list[dict[str, Any]]:
    """Cluster detections by horizontal position into one entry per person.

    In the shots this is for — an interview, a podcast desk — people stay put,
    so their horizontal position is a stable identity. That sidesteps full
    multi-object tracking, and when it is wrong (someone walks across frame)
    the clusters blur together and the caller finds no clear speaker, which is
    the safe outcome.
    """
    people: list[dict[str, Any]] = []
    for when, boxes in detections:
        for box in boxes:
            centre = box["x"] + box["w"] / 2
            match = None
            for person in people:
                if abs(person["x"] - centre) <= SAME_FACE_X:
                    match = person
                    break
            if match is None:
                people.append({"x": centre, "seen": {when: box}})
                continue
            # Running mean keeps the cluster centred as the person shifts.
            seen = match["seen"]
            seen[when] = box
            match["x"] = (match["x"] * (len(seen) - 1) + centre) / len(seen)

    frames = len({when for when, _ in detections}) or 1
    people = [p for p in people if len(p["seen"]) / frames >= MIN_PRESENCE]
    return sorted(people, key=lambda p: p["x"])


def speech_mask(
    words: Sequence[dict[str, Any]],
    span: timeline.Span,
    times: Sequence[float],
) -> list[bool]:
    """For each sampled moment, was a word being spoken then?

    Whisper's word timings are a much cleaner speech signal than audio energy:
    background music, laughter and room tone all raise the volume without
    anyone talking.
    """
    start = float(span[0])
    intervals = [(float(w["start"]) - start, float(w["end"]) - start)
                 for w in words
                 if float(w["end"]) > start and float(w["start"]) < float(span[1])]
    mask: list[bool] = []
    for when in times:
        mask.append(any(a - 0.05 <= when <= b + 0.05 for a, b in intervals))
    return mask


def _patch(grey: Any, box: dict[str, Any], region: str) -> Any | None:
    """A small normalised crop of either the mouth or the whole face."""
    import numpy as np

    width, height = grey.size
    corners = box.get("mouth") or []
    if region == "mouth" and len(corners) == 2:
        # Anchor on the detector's own mouth corners: a fixed fraction of the
        # face box drifts off the mouth as soon as someone tilts their head.
        (rx, ry), (lx, ly) = corners
        cx, cy = (rx + lx) / 2 * width, (ry + ly) / 2 * height
        reach = max(abs(lx - rx) * width * MOUTH_SCALE, 6.0)
        left, right, top, bottom = cx - reach, cx + reach, cy - reach * 0.8, cy + reach * 0.8
    elif region == "mouth":
        left = max(box["x"], 0.0) * width
        right = min(box["x"] + box["w"], 1.0) * width
        top = (box["y"] + box["h"] * 0.55) * height
        bottom = min(box["y"] + box["h"], 1.0) * height
    else:
        left, right = max(box["x"], 0.0) * width, min(box["x"] + box["w"], 1.0) * width
        top, bottom = max(box["y"], 0.0) * height, min(box["y"] + box["h"], 1.0) * height

    crop = (int(max(left, 0)), int(max(top, 0)),
            int(min(right, width)), int(min(bottom, height)))
    if crop[2] - crop[0] < 4 or crop[3] - crop[1] < 4:
        return None
    return np.asarray(grey.crop(crop).resize((32, 16)), dtype=np.float32)


def _motion(
    person: dict[str, Any],
    frames: Sequence[tuple[float, Path]],
) -> dict[str, dict[float, float]]:
    """Per-frame change in this person's mouth, and in their face as a whole.

    Both are needed. Mouth movement alone cannot tell articulation from a
    listener nodding along, because a nod drags the mouth across the frame
    too; dividing one by the other is what separates them.
    """
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return {"mouth": {}, "face": {}}

    series: dict[str, list[tuple[float, Any]]] = {"mouth": [], "face": []}
    for when, path in frames:
        box = person["seen"].get(when)
        if box is None:
            continue
        try:
            with Image.open(path) as img:
                grey = img.convert("L")
                for region in ("mouth", "face"):
                    patch = _patch(grey, box, region)
                    if patch is not None:
                        series[region].append((when, patch))
        except Exception:      # noqa: BLE001
            continue

    out: dict[str, dict[float, float]] = {"mouth": {}, "face": {}}
    for region, patches in series.items():
        for (_t0, a), (t1, b) in zip(patches, patches[1:]):
            # Normalise by brightness so a face in shadow is not read as still.
            scale = float(a.mean()) or 1.0
            out[region][t1] = float(np.abs(b - a).mean()) / scale
    return out


def score_people(
    people: Sequence[dict[str, Any]],
    frames: Sequence[tuple[float, Path]],
    mask: Sequence[bool],
    times: Sequence[float],
) -> list[dict[str, Any]]:
    """How much each person's mouth articulates while words are being spoken.

    Scored as mouth movement damped by head movement, over the frames where
    Whisper says a word was being said. An earlier version subtracted each
    face's own motion during silence instead; on continuous speech that
    baseline is a handful of noisy frames, and it ranked a *frozen* face above
    a talking one, because the talker moves during the gaps too.
    """
    speaking = {t for t, on in zip(times, mask) if on}
    scored: list[dict[str, Any]] = []
    for person in people:
        motion = _motion(person, frames)
        mouth = [v for t, v in motion["mouth"].items() if t in speaking]
        face = [v for t, v in motion["face"].items() if t in speaking]
        if not mouth:
            continue
        articulation = sum(mouth) / len(mouth)
        head = sum(face) / len(face) if face else 0.0
        scored.append({
            "x": person["x"],
            # +1 keeps a still face at zero rather than dividing by noise.
            "score": articulation / (1.0 + head),
            "seen": person["seen"],
        })
    return scored


def _windows(length: float, *, step: float = WINDOW_SECONDS) -> list[timeline.Span]:
    """Tile the span into equal slices to reconsider the speaker in."""
    if length <= 0:
        return []
    edges = []
    position = 0.0
    while position < length - 1e-6:
        edges.append((position, min(position + step, length)))
        position += step
    return edges


def find_speaker_track(
    source: str | Path,
    span: timeline.Span,
    words: Sequence[dict[str, Any]],
    *,
    work_dir: str | Path,
    model: str | Path | None = None,
) -> list[dict[str, Any]] | None:
    """Crop centres that follow whoever is talking, or None to fall back.

    None means "no better answer than per-shot framing" — one face, no model,
    or two faces with nothing to choose between them.
    """
    model = model or model_file()
    if model is None:
        return None

    frame_dir = Path(work_dir) / "_speakers"
    try:
        frames = extract_sequence(source, span, frame_dir)
        if len(frames) < 4:
            return None
        detections = detect_faces(frames, model)
        people = group_people(detections)
        if len(people) < 2:
            return None        # one person: per-shot framing is already right

        times = [when for when, _ in frames]
        length = float(span[1]) - float(span[0])
        track: list[dict[str, Any]] = []

        for a, b in _windows(length):
            inside = [(t, p) for t, p in frames if a <= t <= b]
            if len(inside) < 3:
                continue
            window_times = [t for t, _ in inside]
            mask = speech_mask(words, span, window_times)
            if sum(mask) < MIN_SPEECH_FRAMES:
                continue       # nobody is talking, so nobody to follow
            scored = score_people(people, inside, mask, window_times)
            if len(scored) < 2:
                continue
            scored.sort(key=lambda s: s["score"], reverse=True)
            best, runner_up = scored[0], scored[1]
            if best["score"] <= 0:
                continue
            margin = best["score"] - max(runner_up["score"], 0.0)
            if margin < MIN_MARGIN * max(abs(runner_up["score"]), 1e-6):
                continue       # too close to call, so do not move the frame
            track.append({"start": a, "end": b, "x": best["x"]})

        return _consolidate(track, length)
    finally:
        shutil.rmtree(frame_dir, ignore_errors=True)


def _merge_same(track: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fuse neighbouring entries that name the same person."""
    merged: list[dict[str, Any]] = []
    for entry in track:
        if merged and abs(merged[-1]["x"] - entry["x"]) < SAME_FACE_X:
            merged[-1]["end"] = entry["end"]
        else:
            merged.append(dict(entry))
    return merged


def _consolidate(
    track: Sequence[dict[str, Any]],
    length: float,
) -> list[dict[str, Any]] | None:
    """Merge neighbouring turns, drop flickers, and cover the whole span."""
    if not track:
        return None

    merged = _merge_same(track)

    # A turn too short to hold is detector noise; give its time to the turn
    # before it rather than flicking the frame across and straight back.
    held: list[dict[str, Any]] = []
    for entry in merged:
        if (entry["end"] - entry["start"] < MIN_HOLD_SECONDS) and held:
            held[-1]["end"] = entry["end"]
        else:
            held.append(dict(entry))
    if len(held) > 1 and held[0]["end"] - held[0]["start"] < MIN_HOLD_SECONDS:
        held[1]["start"] = held[0]["start"]
        held.pop(0)

    # Dropping a flicker leaves the same speaker either side of where it was,
    # so merge again — otherwise the renderer steps the crop from a position
    # to itself.
    held = _merge_same(held)

    # The crop must be defined for every frame, so each turn runs until the
    # next one starts, and the first covers the run-in from zero.
    held[0]["start"] = 0.0
    for current, following in zip(held, held[1:]):
        current["end"] = following["start"]
    held[-1]["end"] = length

    # A single entry is still worth returning: one person talking throughout a
    # two-shot is precisely when the crop should sit on them, rather than
    # splitting the difference between two faces the way per-shot framing
    # would.
    return held
