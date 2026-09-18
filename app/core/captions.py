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

#: Fade applied when a phrase appears and disappears, in milliseconds.
PHRASE_FADE_MS = 90

#: Shortest time a word may hold the highlight. Whisper gives function words
#: like "in" or "the" as little as 60ms — two frames at 30fps — and a highlight
#: that jumps that fast reads as flicker rather than as following the speech.
#: Slots below this are absorbed into the neighbouring word's.
MIN_HIGHLIGHT_SECONDS = 0.13

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
    """One on-screen phrase, with the word currently being spoken marked.

    `first`/`last` mark the phrase's outer edges. Only those get a fade: a
    phrase is redrawn once per word, so fading every event makes the whole
    line pulse on every syllable.
    """

    start: float
    end: float
    tokens: list[str] = field(default_factory=list)
    active: int = 0
    first: bool = True
    last: bool = True

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


def _slots(group: Sequence[dict[str, Any]]) -> list[tuple[float, float, int]]:
    """`(start, end, highlighted_word)` per word, tiled with no gaps or overlaps.

    Each word holds the highlight until the next one starts. Slots too short to
    perceive are absorbed by a neighbour, so the highlight moves at a readable
    pace instead of strobing through short words.
    """
    group_end = max(float(w["end"]) for w in group)
    raw: list[tuple[float, float, int]] = []
    cursor = float(group[0]["start"])
    for i in range(len(group)):
        end = float(group[i + 1]["start"]) if i + 1 < len(group) else group_end
        end = max(end, cursor)
        raw.append((cursor, end, i))
        cursor = end

    merged: list[tuple[float, float, int]] = []
    for start, end, active in raw:
        if end - start < MIN_HIGHLIGHT_SECONDS and merged:
            # Hold the previous word's highlight through this one.
            prev_start, _prev_end, prev_active = merged[-1]
            merged[-1] = (prev_start, end, prev_active)
            continue
        merged.append((start, end, active))

    # A too-short opening slot has no predecessor, so give its time to the next.
    if len(merged) > 1 and merged[0][1] - merged[0][0] < MIN_HIGHLIGHT_SECONDS:
        _start, _end, _active = merged.pop(0)
        nxt_start, nxt_end, nxt_active = merged[0]
        merged[0] = (_start, nxt_end, nxt_active)
    return merged


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
        # Slots tile exactly. Deriving a start from the *previous* word's end
        # instead would make consecutive events overlap by the silence between
        # the words, and two simultaneous events render as two stacked lines.
        slots = _slots(group)
        for n, (start, end, active) in enumerate(slots):
            ev_start = max(0.0, min(start, duration))
            ev_end = max(ev_start + 0.04, min(end, duration))
            if ev_start >= duration:
                continue
            events.append(CaptionEvent(
                start=round(ev_start, 3), end=round(ev_end, 3),
                tokens=list(tokens), active=active,
                first=n == 0, last=n == len(slots) - 1,
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
                # Colour only. Scaling the active word changes the line's width,
                # which re-centres it and makes the text jump sideways on every
                # word — the single worst source of caption jitter.
                parts.append(f"{{\\c{ACTIVE_COLOUR}}}{text}{{\\c{IDLE_COLOUR}}}")
            else:
                parts.append(text)

        # Fade in when the phrase appears and out when it leaves, never between
        # its words: consecutive events carry identical text, so a mid-phrase
        # fade reads as the line blinking rather than a highlight moving.
        fade_in = PHRASE_FADE_MS if event.first else 0
        fade_out = PHRASE_FADE_MS if event.last else 0
        prefix = (
            f"{{\\fad({fade_in},{fade_out})}}" if (fade_in or fade_out) else ""
        )
        body = prefix + " ".join(parts)
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
