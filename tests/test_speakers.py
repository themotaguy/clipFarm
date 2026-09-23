"""Following whoever is talking in a shot that holds more than one face."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from app.core import facedetect, media, speakers


# --- grouping detections into people ---

def _frames(*per_frame: list[float]) -> list[tuple[float, list[dict]]]:
    """Detections given as the x-centre of each face in each frame."""
    out = []
    for i, centres in enumerate(per_frame):
        boxes = [{"x": c - 0.05, "y": 0.2, "w": 0.1, "h": 0.2, "score": 0.9}
                 for c in centres]
        out.append((i * 0.2, boxes))
    return out


def test_two_stable_positions_become_two_people():
    detections = _frames([0.25, 0.75], [0.26, 0.74], [0.25, 0.76], [0.24, 0.75])
    people = speakers.group_people(detections)
    assert len(people) == 2
    assert people[0]["x"] < 0.5 < people[1]["x"], "people are sorted left to right"


def test_a_face_seen_once_is_not_a_participant():
    """Someone crossing the background is not part of the conversation."""
    detections = _frames([0.25], [0.25], [0.25], [0.25], [0.25, 0.9])
    people = speakers.group_people(detections)
    assert [round(p["x"], 2) for p in people] == [0.25]


def test_one_person_is_one_cluster_despite_small_drift():
    detections = _frames([0.50], [0.52], [0.48], [0.51], [0.49])
    assert len(speakers.group_people(detections)) == 1


def test_no_detections_means_no_people():
    assert speakers.group_people([]) == []


# --- the speech signal ---

def test_speech_mask_marks_only_spoken_moments():
    words = [{"start": 10.0, "end": 10.5}, {"start": 12.0, "end": 12.4}]
    mask = speakers.speech_mask(words, (10.0, 14.0), [0.2, 1.0, 2.2, 3.5])
    assert mask == [True, False, True, False]


def test_speech_mask_ignores_words_outside_the_span():
    words = [{"start": 1.0, "end": 2.0}, {"start": 50.0, "end": 51.0}]
    assert speakers.speech_mask(words, (10.0, 14.0), [0.5, 2.0]) == [False, False]


def test_speech_mask_without_words_is_all_silent():
    assert speakers.speech_mask([], (0.0, 5.0), [0.0, 1.0]) == [False, False]


# --- analysis windows ---

def test_windows_tile_the_whole_span():
    windows = speakers._windows(7.0, step=2.0)
    assert windows[0][0] == 0.0
    assert windows[-1][1] == 7.0
    for a, b in zip(windows, windows[1:]):
        assert a[1] == b[0], "gap between windows"


def test_windows_of_an_empty_span():
    assert speakers._windows(0.0) == []


# --- consolidating turns ---

def test_consolidate_merges_the_same_speaker():
    track = [{"start": 0.0, "end": 2.0, "x": 0.3},
             {"start": 2.0, "end": 4.0, "x": 0.31},
             {"start": 4.0, "end": 6.0, "x": 0.3}]
    out = speakers._consolidate(track, 6.0)
    assert len(out) == 1
    assert out[0] == {"start": 0.0, "end": 6.0, "x": 0.3}


def test_consolidate_keeps_a_real_change_of_speaker():
    track = [{"start": 0.0, "end": 4.0, "x": 0.25},
             {"start": 4.0, "end": 8.0, "x": 0.75}]
    out = speakers._consolidate(track, 8.0)
    assert [round(e["x"], 2) for e in out] == [0.25, 0.75]
    assert out[0]["start"] == 0.0 and out[-1]["end"] == 8.0


def test_consolidate_swallows_a_flicker():
    """A sub-second jump to the other face and back is detector noise."""
    track = [{"start": 0.0, "end": 4.0, "x": 0.25},
             {"start": 4.0, "end": 4.4, "x": 0.75},
             {"start": 4.4, "end": 8.0, "x": 0.25}]
    out = speakers._consolidate(track, 8.0)
    assert len(out) == 1, f"frame flicked across for 0.4s: {out}"


def test_consolidate_covers_every_moment():
    track = [{"start": 1.5, "end": 3.0, "x": 0.25},
             {"start": 5.0, "end": 7.0, "x": 0.75}]
    out = speakers._consolidate(track, 9.0)
    assert out[0]["start"] == 0.0, "the run-in before the first word is uncovered"
    assert out[-1]["end"] == 9.0, "the tail after the last word is uncovered"
    for a, b in zip(out, out[1:]):
        assert a["end"] == b["start"], "gap in the crop timeline"


def test_consolidate_of_nothing_is_none():
    assert speakers._consolidate([], 10.0) is None


def test_a_single_constant_speaker_is_still_a_track():
    """Two faces, one talking throughout: crop to them, do not split the
    difference the way per-shot framing would."""
    out = speakers._consolidate([{"start": 0.0, "end": 9.0, "x": 0.28}], 9.0)
    assert out is not None and len(out) == 1
    assert out[0]["x"] == 0.28


# --- the model ---

def test_model_file_is_cached_and_not_refetched(tmp_path, monkeypatch):
    cached = tmp_path / speakers.MODEL_NAME
    cached.write_bytes(b"pretend model")

    def explode(*_a, **_k):
        raise AssertionError("re-downloaded a model that was already cached")

    monkeypatch.setattr(speakers.urllib.request, "urlopen", explode)
    assert speakers.model_file(tmp_path) == cached


def test_model_file_returns_none_when_it_cannot_be_fetched(tmp_path, monkeypatch):
    """No model must mean no speaker tracking, not a failed render."""
    def explode(*_a, **_k):
        raise OSError("offline")

    monkeypatch.setattr(speakers.urllib.request, "urlopen", explode)
    assert speakers.model_file(tmp_path) is None
    assert not (tmp_path / speakers.MODEL_NAME).exists()


# --- the detection subprocess ---

pytestmark_ffmpeg = pytest.mark.skipif(
    not media.has_ffmpeg(), reason="ffmpeg not on PATH"
)


@pytestmark_ffmpeg
def test_worker_reports_no_faces_in_footage_without_any(tmp_path):
    """Exercises the real subprocess contract end to end, faces or not."""
    model = speakers.model_file()
    if model is None:
        pytest.skip("face model not available offline")

    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    media.run([
        media.binary("ffmpeg"), "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc2=s=640x360:r=5:d=1",
        str(frame_dir / "f-%05d.jpg"),
    ])
    assert list(frame_dir.glob("f-*.jpg")), "fixture produced no frames"

    proc = subprocess.run(
        [sys.executable, "-m", "app.core.facedetect", str(model), str(frame_dir)],
        capture_output=True, text=True, check=False,
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["frames"], "worker reported no frames at all"
    assert all(entry["boxes"] == [] for entry in payload["frames"])


def test_worker_on_an_empty_directory(tmp_path):
    assert facedetect.detect_dir("unused.onnx", tmp_path) == []


def test_detect_faces_without_frames_is_empty():
    assert speakers.detect_faces([], "unused.onnx") == []
