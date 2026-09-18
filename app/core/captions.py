"""Word-timed captions for burned-in subtitles.

Whisper gives per-word timestamps, so instead of static blocks we emit one
event per word: the phrase stays on screen while the word being spoken is
recoloured. That is the look short-form audiences expect.

The word grouping and highlighting live in `build_events`, which both burn-in
backends share — libass (`subtitles=`) and the Pillow renderer in
`pngcaptions` — so captions look the same however they get drawn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from app.core import timeline

MAX_WORDS_PER_GROUP = 4
MAX_CHARS_PER_GROUP = 24

# ASS colours are &HBBGGRR (not RGB).
IDLE_COLOUR = "&H00FFFFFF&"      # white
ACTIVE_COLOUR = "&H0047E3FF&"    # amber

#: RGB equivalents, for backends that draw the text themselves.
IDLE_RGB = (255, 255, 255)
ACTIVE_RGB = (255, 227, 71)

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


@dataclass
class CaptionEvent:
    """One on-screen phrase, with the word currently being spoken marked."""

    start: float
    end: float
    tokens: list[str] = field(default_factory=list)
    active: int = 0

    @property
    def text(self) -> str:
        return " ".join(self.tokens)


def collect_words(
    segments: Iterable[dict[str, Any]],
    spans: Sequence[timeline.Span],
) -> list[dict[str, Any]]:
    """Words inside `spans`, with times rebased onto the clip timeline."""
    return timeline.words_in_spans(segments, timeline.normalize(spans))


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


def build_events(
    segments: Iterable[dict[str, Any]],
    spans: Sequence[timeline.Span],
    *,
    duration: float | None = None,
) -> list[CaptionEvent]:
    """Word-by-word caption events on the clip timeline.

    Events are tiled edge to edge within a phrase so it never flickers between
    words, and clamped to the clip duration.
    """
    spans = timeline.normalize(spans)
    words = collect_words(segments, spans)
    if duration is None:
        duration = timeline.total_duration(spans)
    duration = max(0.1, duration)

    events: list[CaptionEvent] = []
    for group in _group(words):
        tokens = [w["word"] for w in group]
        group_end = max(w["end"] for w in group)
        # Each word holds the highlight until the next one starts, so events
        # tile exactly: the phrase never flickers and never doubles up.
        #
        # Deriving the start from the *previous* word's end instead would make
        # consecutive events overlap by the silence between the words, and two
        # simultaneous events render as two stacked lines of subtitles.
        cursor = float(group[0]["start"])
        for i, _word in enumerate(group):
            ev_end = (
                float(group[i + 1]["start"]) if i + 1 < len(group) else group_end
            )
            ev_start = max(0.0, min(cursor, duration))
            ev_end = max(ev_start + 0.04, min(ev_end, duration))
            cursor = ev_end
            if ev_start >= duration:
                continue
            events.append(CaptionEvent(
                start=round(ev_start, 3), end=round(ev_end, 3),
                tokens=list(tokens), active=i,
            ))
    return events


def build_ass(
    segments: Iterable[dict[str, Any]],
    spans: Sequence[timeline.Span],
    *,
    width: int = 1080,
    height: int = 1920,
    font: str | None = None,
    duration: float | None = None,
) -> str:
    """Render an ASS subtitle document for the clip made of `spans`."""
    import config

    font = font or config.CAPTION_FONT
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

    events = build_events(segments, spans, duration=duration)
    if not events:
        return header

    lines: list[str] = []
    for event in events:
        parts = []
        for j, token in enumerate(event.tokens):
            text = _escape(token)
            if j == event.active:
                parts.append(f"{{\\c{ACTIVE_COLOUR}\\fscx108\\fscy108}}{text}"
                             f"{{\\c{IDLE_COLOUR}\\fscx100\\fscy100}}")
            else:
                parts.append(text)
        # A fixed 60ms in/out fade eats most of a short event: on fast speech a
        # word can hold the highlight for only ~140ms, which would leave it
        # fading for 120ms of that and looking washed out.
        fade = max(10, min(60, int((event.end - event.start) * 1000 / 4)))
        body = f"{{\\fad({fade},{fade})}}" + " ".join(parts)
        lines.append(
            f"Dialogue: 0,{_ts(event.start)},{_ts(event.end)},Pop,,0,0,0,,{body}"
        )

    return header + "\n".join(lines) + "\n"


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
