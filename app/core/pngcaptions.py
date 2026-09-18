"""Caption burn-in that does not need libass.

Homebrew's plain `ffmpeg` formula ships without libass, so the `subtitles`
filter simply does not exist there and the usual ASS burn-in fails outright.
Rather than leave captions off — they are the whole point of a short-form clip —
we draw each caption phrase ourselves with Pillow and composite the resulting
images with ffmpeg's `overlay`, which every build has.

One PNG is produced per caption event (a phrase with one word highlighted), and
each is enabled for exactly its time range. Visually this matches the libass
path: same grouping, same colours, same placement.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Sequence

from app.core.captions import (
    ACTIVE_RGB,
    IDLE_RGB,
    CaptionEvent,
)

#: Above this many overlays the filtergraph gets unwieldy, so captions fall
#: back to one image per phrase (no per-word highlight) to keep the count down.
MAX_OVERLAYS = 320

#: Where to look for the configured caption font, by family name.
FONT_DIRS = (
    "/System/Library/Fonts/Supplemental",
    "/System/Library/Fonts",
    "/Library/Fonts",
    str(Path.home() / "Library/Fonts"),
    "/usr/share/fonts/truetype/dejavu",
    "/usr/share/fonts/truetype",
    "/usr/share/fonts",
)

#: Tried in order when the configured font cannot be found.
FONT_FALLBACKS = (
    "Arial Black.ttf", "Arial Bold.ttf", "Arial.ttf",
    "HelveticaNeue.ttc", "Helvetica.ttc",
    "DejaVuSans-Bold.ttf", "DejaVuSans.ttf",
)


def available() -> bool:
    """Whether Pillow is installed, i.e. whether we can draw captions at all."""
    try:
        import PIL  # noqa: F401
    except ImportError:
        return False
    return True


def find_font_file(name: str | None = None) -> str | None:
    """Locate a TrueType file for a font family name, or a sane fallback."""
    import config

    if config.__dict__.get("CAPTION_FONT_FILE"):       # explicit override
        override = Path(config.CAPTION_FONT_FILE)
        if override.is_file():
            return str(override)

    wanted = (name or config.CAPTION_FONT or "").strip()
    candidates: list[str] = []
    if wanted:
        # "Arial Black" -> "Arial Black.ttf", "ArialBlack.ttf"
        candidates += [f"{wanted}.ttf", f"{wanted}.ttc",
                       f"{wanted.replace(' ', '')}.ttf"]
    candidates += list(FONT_FALLBACKS)

    for filename in candidates:
        for directory in FONT_DIRS:
            path = Path(directory) / filename
            if path.is_file():
                return str(path)

    # Last resort: anything TrueType we can find.
    for directory in FONT_DIRS:
        d = Path(directory)
        if not d.is_dir():
            continue
        for path in sorted(d.glob("*.ttf")):
            return str(path)
    return None


def _load_font(size: int):
    from PIL import ImageFont

    path = find_font_file()
    if path:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default(size)


def _advance(draw, text: str, font) -> int:
    """Horizontal advance of `text`, i.e. where the next glyph would start.

    `textlength` is the right measure for laying words side by side; a bounding
    box would ignore trailing space and drift on every word.
    """
    return int(round(draw.textlength(text, font=font)))


def _line_height(font) -> int:
    """Ascent + descent, so descenders are never clipped off the canvas.

    Sizing from a bounding box delta loses both the ascender offset and the
    descent, which cuts the bottom off glyphs like 'p' and 'g'.
    """
    try:
        ascent, descent = font.getmetrics()
        return int(ascent + descent)
    except AttributeError:      # bitmap fallback font
        return int(getattr(font, "size", 32) * 1.3)


def _wrap(tokens: Sequence[str], draw, font, max_width: int) -> list[list[int]]:
    """Split token indices into lines that each fit `max_width`."""
    lines: list[list[int]] = []
    current: list[int] = []
    for idx, token in enumerate(tokens):
        trial = " ".join([*(tokens[i] for i in current), token])
        if current and _advance(draw, trial, font) > max_width:
            lines.append(current)
            current = [idx]
        else:
            current.append(idx)
    if current:
        lines.append(current)
    return lines


def _collapse(events: Sequence[CaptionEvent]) -> list[CaptionEvent]:
    """One event per phrase, dropping the per-word highlight."""
    collapsed: list[CaptionEvent] = []
    for event in events:
        if collapsed and collapsed[-1].tokens == event.tokens:
            collapsed[-1].end = event.end
            continue
        collapsed.append(CaptionEvent(
            start=event.start, end=event.end, tokens=list(event.tokens), active=-1
        ))
    return collapsed


def render_events(
    events: Sequence[CaptionEvent],
    out_dir: str | Path,
    *,
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    """Draw each caption event to a PNG. Returns overlay descriptors.

    Images are cropped to the text, so the filtergraph only carries the pixels
    it needs rather than a full-frame RGBA plane per event.
    """
    from PIL import Image, ImageDraw

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if len(events) > MAX_OVERLAYS:
        events = _collapse(events)

    size = max(40, int(height * 0.048))
    font = _load_font(size)
    stroke = max(3, int(size * 0.09))
    max_text_width = int(width * 0.82)
    line_gap = int(size * 0.22)

    # A scratch canvas just for text measurement.
    probe = ImageDraw.Draw(Image.new("RGBA", (8, 8)))

    overlays: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        tokens = event.tokens
        if not tokens:
            continue

        rows = _wrap(tokens, probe, font, max_text_width)
        row_widths = [
            _advance(probe, " ".join(tokens[i] for i in row), font) for row in rows
        ]
        row_h = _line_height(font)
        # The stroke grows the glyphs in every direction, so pad by it on all sides.
        pad = stroke + 4
        canvas_w = max(row_widths) + 2 * pad
        canvas_h = len(rows) * row_h + (len(rows) - 1) * line_gap + 2 * pad

        image = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)

        y = pad
        for row, row_w in zip(rows, row_widths):
            x = (canvas_w - row_w) // 2
            for idx in row:
                token = tokens[idx]
                colour = ACTIVE_RGB if idx == event.active else IDLE_RGB
                draw.text(
                    (x, y), token, font=font, fill=(*colour, 255),
                    stroke_width=stroke, stroke_fill=(16, 16, 16, 235),
                )
                x += _advance(probe, token + " ", font)
            y += row_h + line_gap

        path = out_dir / f"cap-{index:05d}.png"
        image.save(path)
        overlays.append({
            "path": str(path),
            "start": float(event.start),
            "end": float(event.end),
            "width": canvas_w,
            "height": canvas_h,
        })

    return overlays


def cleanup(overlays: Sequence[dict[str, Any]]) -> None:
    """Remove the intermediate PNGs once ffmpeg has consumed them."""
    for overlay in overlays:
        directory = Path(overlay["path"]).parent
        shutil.rmtree(directory, ignore_errors=True)
        return


def input_args(overlays: Sequence[dict[str, Any]]) -> list[str]:
    """ffmpeg `-i` arguments for the caption images, in order."""
    args: list[str] = []
    for overlay in overlays:
        args += ["-loop", "1", "-t", f"{max(overlay['end'], 0.1):.3f}",
                 "-i", overlay["path"]]
    return args


def filter_chain(
    overlays: Sequence[dict[str, Any]],
    *,
    first_input: int,
    in_label: str,
    out_label: str,
    height: int,
) -> str:
    """Chain one `overlay` per caption event onto `in_label`.

    Each image is enabled only for its own time range, so exactly one phrase is
    ever visible.
    """
    if not overlays:
        return f"[{in_label}]null[{out_label}]"

    # Match the libass style's vertical margin so both backends agree.
    margin_v = int(height * 0.16)
    steps: list[str] = []
    current = in_label
    for i, overlay in enumerate(overlays):
        idx = first_input + i
        nxt = out_label if i == len(overlays) - 1 else f"cap{i}"
        y = max(0, height - margin_v - overlay["height"])
        steps.append(
            f"[{current}][{idx}:v]overlay=x=(W-w)/2:y={y}:"
            f"enable='between(t,{overlay['start']:.3f},{overlay['end']:.3f})'"
            f"[{nxt}]"
        )
        current = nxt
    return ";".join(steps)
