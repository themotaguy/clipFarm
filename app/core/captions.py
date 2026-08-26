"""Word-timed ASS subtitles for burned-in captions.

Whisper gives per-word timestamps, so instead of static blocks we emit one event
per word: the phrase stays on screen while the word being spoken is recoloured.
That is the look short-form audiences expect.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

MAX_WORDS_PER_GROUP = 4
MAX_CHARS_PER_GROUP = 24

# ASS colours are &HBBGGRR (not RGB).
IDLE_COLOUR = "&H00FFFFFF&"      # white
ACTIVE_COLOUR = "&H0047E3FF&"    # amber

ASS_HEADER = """\
[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 2
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Pop,{font},{size},&H00FFFFFF,&H000000FF,&H00101010,&HA0000000,-1,0,0,0,100,100,1,0,1,{outline},2,2,{margin_h},{margin_h},{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def collect_words(
    segments: Sequence[dict[str, Any]],
    start: float,
    end: float,
) -> list[dict[str, Any]]:
    """Words falling inside [start, end], with times rebased to the clip."""
    words: list[dict[str, Any]] = []
    for seg in segments:
        if seg["end"] <= start or seg["start"] >= end:
            continue
        for w in seg.get("words") or []:
            if w["end"] <= start or w["start"] >= end:
                continue
            token = (w.get("word") or "").strip()
            if not token:
                continue
            words.append({
                "word": token,
                "start": max(0.0, float(w["start"]) - start),
                "end": max(0.0, min(float(w["end"]), end) - start),
            })
    return words


def _group(words: Sequence[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Break the word stream into short on-screen phrases."""
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    chars = 0
    for w in words:
        token = w["word"]
        too_many = len(current) >= MAX_WORDS_PER_GROUP
        too_wide = chars + len(token) + 1 > MAX_CHARS_PER_GROUP
        # A sentence-ending word closes the phrase so captions break naturally.
        if current and (too_many or too_wide):
            groups.append(current)
            current, chars = [], 0
        current.append(w)
        chars += len(token) + 1
        if token.endswith((".", "!", "?")) and len(current) >= 2:
            groups.append(current)
            current, chars = [], 0
    if current:
        groups.append(current)
    return groups


def build_ass(
    segments: Sequence[dict[str, Any]],
    start: float,
    end: float,
    *,
    width: int = 1080,
    height: int = 1920,
    font: str = "Arial Black",
) -> str:
    """Render an ASS subtitle document for the clip spanning [start, end]."""
    words = collect_words(segments, start, end)
    duration = max(0.1, end - start)

    size = max(40, int(height * 0.048))
    header = ASS_HEADER.format(
        width=width,
        height=height,
        font=font,
        size=size,
        outline=max(3, int(size * 0.09)),
        margin_h=int(width * 0.09),
        margin_v=int(height * 0.16),
    )
    if not words:
        return header

    events: list[str] = []
    for group in _group(words):
        group_end = max(w["end"] for w in group)
        for i, word in enumerate(group):
            ev_start = word["start"] if i == 0 else group[i - 1]["end"]
            # Tile events edge to edge so the phrase never flickers between words.
            ev_end = group[i + 1]["start"] if i + 1 < len(group) else group_end
            ev_start = max(0.0, min(ev_start, duration))
            ev_end = max(ev_start + 0.04, min(ev_end, duration))

            parts = []
            for j, other in enumerate(group):
                text = _escape(other["word"])
                if j == i:
                    parts.append(f"{{\\c{ACTIVE_COLOUR}\\fscx108\\fscy108}}{text}"
                                 f"{{\\c{IDLE_COLOUR}\\fscx100\\fscy100}}")
                else:
                    parts.append(text)
            body = "{\\fad(60,60)}" + " ".join(parts)
            events.append(
                f"Dialogue: 0,{_ts(ev_start)},{_ts(ev_end)},Pop,,0,0,0,,{body}"
            )

    return header + "\n".join(events) + "\n"


def write_ass(path: str | Path, content: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _escape(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("\n", " ")
    )


def _ts(seconds: float) -> str:
    """ASS timestamps are H:MM:SS.cc with centisecond precision."""
    cs = int(round(max(0.0, seconds) * 100))
    h, cs = divmod(cs, 360_000)
    m, cs = divmod(cs, 6_000)
    s, cs = divmod(cs, 100)
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"
