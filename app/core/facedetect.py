"""Face detection worker, run as a subprocess.

This exists as a separate process on purpose, for two reasons found the hard
way on this machine:

* **OpenCV ships its own FFmpeg libraries**, and they collide with the ones
  PyAV (a faster-whisper dependency) loads: "Class AVFFrameReceiver is
  implemented in both ... This may cause spurious casting failures and
  mysterious crashes." Keeping `cv2` out of the main interpreter avoids the
  question entirely.
* **Native detectors can abort the process, not raise.** MediaPipe 1.0.1 dies
  with a C++ `CHECK` failure on macOS ("Service is unavailable") that no
  `try/except` can catch. In a subprocess the worst case is a non-zero exit
  and a render that falls back to per-shot framing.

Usage: `python -m app.core.facedetect <model.onnx> <frame-dir>`, printing one
JSON object on stdout. Every box is normalised to 0..1 of the frame size so
the caller never has to care what resolution the frames were sampled at.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

#: Detections below this confidence are dropped.
SCORE_THRESHOLD = 0.6
NMS_THRESHOLD = 0.3


def detect_dir(model: str | Path, frame_dir: str | Path) -> list[dict[str, Any]]:
    """Detect faces in every `f-*.jpg` under `frame_dir`, in filename order."""
    import cv2

    frames = sorted(Path(frame_dir).glob("f-*.jpg"))
    if not frames:
        return []

    first = cv2.imread(str(frames[0]))
    if first is None:
        return []
    height, width = first.shape[:2]
    detector = cv2.FaceDetectorYN.create(
        str(model), "", (width, height), SCORE_THRESHOLD, NMS_THRESHOLD, 5000,
    )

    out: list[dict[str, Any]] = []
    for index, path in enumerate(frames):
        image = cv2.imread(str(path))
        if image is None:
            continue
        if image.shape[0] != height or image.shape[1] != width:
            height, width = image.shape[:2]
            detector.setInputSize((width, height))
        _count, faces = detector.detect(image)
        boxes = []
        for face in faces if faces is not None else []:
            values = [float(v) for v in face]
            # YuNet: x, y, w, h, then five landmarks (eyes, nose, both mouth
            # corners) as x/y pairs, then the score.
            boxes.append({
                "x": values[0] / width,
                "y": values[1] / height,
                "w": values[2] / width,
                "h": values[3] / height,
                "mouth": [
                    (values[10] / width, values[11] / height),
                    (values[12] / width, values[13] / height),
                ],
                "score": values[14],
            })
        out.append({"frame": index, "name": path.name, "boxes": boxes})
    return out


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: python -m app.core.facedetect <model> <frame-dir>",
              file=sys.stderr)
        return 2
    try:
        print(json.dumps({"frames": detect_dir(argv[1], argv[2])}))
    except Exception as exc:      # noqa: BLE001 - the caller just falls back
        print(json.dumps({"error": str(exc)}))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
