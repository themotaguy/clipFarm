"""Stage 6 — cut the scored spans into a vertical clip with burned-in captions.

Layout is the standard short-form treatment: the source frame is scaled to fit
the 9:16 canvas and centred over a blurred, darkened copy of itself that fills
the rest, with word-timed captions on top.

A clip may be made of more than one span. When the scorer decides the payoff
only lands if an earlier setup line is included, the spans are cut separately
and concatenated, so one clip can carry context from two places in the video.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any, Callable, Sequence

import config
from app.core import captions as captions_mod
from app.core import boundaries
from app.core import framing
from app.core import media
from app.core import pngcaptions
from app.core import speakers
from app.core import timeline
from app.core import transcribe as transcribe_mod

ProgressFn = Callable[[float], None]

_TIME_RE = re.compile(r"^out_time_(?:us|ms)=(\d+)$")

#: Common audio format for every span, so concat never sees a mismatch.
_AFMT = "aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo"


def _run_ffmpeg(
    args: list[str],
    *,
    duration: float,
    cwd: Path | None = None,
    on_progress: ProgressFn | None = None,
) -> None:
    """Run ffmpeg, translating its -progress stream into a 0..1 fraction."""
    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(cwd) if cwd else None,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        match = _TIME_RE.match(line.strip())
        if match and on_progress and duration > 0:
            # out_time_us is microseconds, out_time_ms is (confusingly) also micro.
            seconds = int(match.group(1)) / 1_000_000
            on_progress(max(0.0, min(1.0, seconds / duration)))
    proc.wait()
    if proc.returncode != 0:
        stderr = (proc.stderr.read() if proc.stderr else "") or ""
        tail = stderr.strip().splitlines()[-12:]
        raise media.MediaError("ffmpeg failed while rendering:\n" + "\n".join(tail))


def caption_backend(burn: bool) -> str:
    """Which caption renderer to use: "libass", "pillow" or "none"."""
    if not burn or config.CAPTION_RENDERER == "none":
        return "none"

    choice = config.CAPTION_RENDERER
    if choice == "libass":
        return "libass" if media.has_filter("subtitles") else "none"
    if choice == "pillow":
        return "pillow" if pngcaptions.available() else "none"

    # auto: libass does real text shaping, so prefer it whenever it exists.
    if media.has_filter("subtitles"):
        return "libass"
    if pngcaptions.available():
        return "pillow"
    return "none"


def _blur_fill(index: int, width: int, height: int, tag: str) -> tuple[str, str]:
    """Letterbox the whole frame over a blurred, darkened copy of itself."""
    label = f"v{tag}"
    chain = (
        f"[{index}:v]split=2[bg{tag}][fg{tag}];"
        f"[bg{tag}]scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},gblur=sigma=20,eq=brightness=-0.12:saturation=1.15[bgb{tag}];"
        f"[fg{tag}]scale={width}:{height}:force_original_aspect_ratio=decrease[fgs{tag}];"
        f"[bgb{tag}][fgs{tag}]overlay=(W-w)/2:(H-h)/2:format=auto,"
        f"setsar=1,format=yuv420p[{label}]"
    )
    return chain, label


def _crop_x_expression(track: Sequence[dict[str, Any]]) -> str:
    """An ffmpeg expression stepping the crop to each shot's subject.

    The steps land exactly on cuts, where the whole picture changes anyway, so
    a hard jump reads as part of the edit rather than as the camera lurching.
    """
    expression = str(int(track[-1]["x"]))
    for i in range(len(track) - 1, 0, -1):
        cut = float(track[i]["start"])
        expression = (f"if(lt(t,{cut:.3f}),{int(track[i - 1]['x'])},{expression})")
    return expression


def _crop_fill(
    index: int,
    width: int,
    height: int,
    tag: str,
    box: dict[str, int] | None,
    track: Sequence[dict[str, Any]] | None = None,
) -> tuple[str, str]:
    """Fill the canvas by cropping — no blurred bars, subject kept in shot."""
    label = f"v{tag}"
    if box and track and len(track) > 1:
        # Quoted so the commas inside the expression are not read as filter
        # separators by the filtergraph parser.
        crop = (f"crop=w={box['width']}:h={box['height']}"
                f":x='{_crop_x_expression(track)}':y={box['y']}")
    elif box:
        crop = (f"crop=w={box['width']}:h={box['height']}"
                f":x={box['x']}:y={box['y']}")
    else:
        # Source is not wider than the target: cover and centre instead.
        crop = (f"scale={width}:{height}:force_original_aspect_ratio=increase,"
                f"crop={width}:{height}")
    chain = (
        f"[{index}:v]{crop},scale={width}:{height},"
        f"setsar=1,format=yuv420p[{label}]"
    )
    return chain, label


def _composite_video(
    index: int,
    width: int,
    height: int,
    tag: str,
    *,
    mode: str = "blur",
    box: dict[str, int] | None = None,
    track: Sequence[dict[str, Any]] | None = None,
) -> tuple[str, str]:
    """Fit one span onto the vertical canvas using the chosen fill mode."""
    if mode == "crop":
        return _crop_fill(index, width, height, tag, box, track)
    return _blur_fill(index, width, height, tag)


def _build_graph(
    spans: Sequence[timeline.Span],
    *,
    has_video: bool,
    width: int,
    height: int,
    subtitle_file: str | None,
    overlays: Sequence[dict[str, Any]],
    color_input: int | None,
    span_offset: int,
    caption_offset: int,
    fill_mode: str = "blur",
    crop_box: dict[str, int] | None = None,
    crop_tracks: Sequence[Sequence[dict[str, Any]]] | None = None,
) -> str:
    """Whole filtergraph: per-span composite, concat, then captions."""
    parts: list[str] = []
    concat_inputs: list[str] = []

    for i, _span in enumerate(spans):
        idx = span_offset + i
        tag = str(i)
        if has_video:
            track = crop_tracks[i] if crop_tracks and i < len(crop_tracks) else None
            chain, vlabel = _composite_video(idx, width, height, tag,
                                             mode=fill_mode, box=crop_box,
                                             track=track)
            parts.append(chain)
            parts.append(f"[{idx}:a]{_AFMT},asetpts=PTS-STARTPTS[a{tag}]")
        else:
            # Audio-only: draw a waveform for this span, and keep the audio.
            parts.append(f"[{idx}:a]asplit=2[aw{tag}][ao{tag}]")
            parts.append(
                f"[aw{tag}]showwaves=s={width}x{height // 4}:mode=cline:rate=30:"
                f"colors=0x38bdf8|0x818cf8,setsar=1[w{tag}]"
            )
            parts.append(f"[ao{tag}]{_AFMT},asetpts=PTS-STARTPTS[a{tag}]")
            vlabel = f"w{tag}"
        concat_inputs.append(f"[{vlabel}][a{tag}]")

    # concat with n=1 is valid, so a single-span clip takes the same path.
    parts.append(
        "".join(concat_inputs) + f"concat=n={len(spans)}:v=1:a=1[vcat][aout]"
    )

    if not has_video:
        # Lay the concatenated waveform over the flat background canvas.
        parts.append(
            f"[{color_input}:v][vcat]overlay=0:(H-h)/2:format=auto,"
            f"setsar=1,format=yuv420p[vbg]"
        )
        stage = "vbg"
    else:
        stage = "vcat"

    if subtitle_file:
        parts.append(f"[{stage}]subtitles={subtitle_file}[vout]")
    elif overlays:
        parts.append(pngcaptions.filter_chain(
            overlays, first_input=caption_offset, in_label=stage,
            out_label="vout", height=height,
        ))
    else:
        parts.append(f"[{stage}]null[vout]")

    return ";".join(parts)


def _subject_track(
    source: str | Path,
    span: timeline.Span,
    words: Sequence[dict[str, Any]],
    *,
    window: float,
    work_dir: str | Path,
) -> list[dict[str, Any]]:
    """Where to centre the crop through one span, shot by shot and speaker by
    speaker.

    Per-shot framing is the base layer and handles every cut. Inside a shot
    long enough to hold a conversation, speaker tracking gets a chance to
    subdivide it further — that is the only thing that helps a locked-off
    two-shot, where no cut ever happens and detail-and-motion scoring lands
    between the two faces.
    """
    track = framing.find_subject_track(
        source, span, window=window,
        samples=config.FRAMING_SAMPLES, work_dir=work_dir,
    )
    if not track or not config.SPEAKER_TRACKING or not words:
        return track

    refined: list[dict[str, Any]] = []
    for entry in track:
        shot = (float(span[0]) + entry["start"], float(span[0]) + entry["end"])
        if shot[1] - shot[0] < speakers.MIN_SHOT_SECONDS:
            refined.append(entry)
            continue
        found = speakers.find_speaker_track(source, shot, words,
                                            work_dir=work_dir)
        if not found:
            refined.append(entry)      # one face, or nothing conclusive
            continue
        for piece in found:
            refined.append({
                "start": entry["start"] + piece["start"],
                "end": entry["start"] + piece["end"],
                "x": piece["x"],
            })
    return refined


def render_clip(
    source: str | Path,
    out_path: str | Path,
    spans: Any,
    segments: Sequence[dict[str, Any]],
    *,
    has_video: bool = True,
    burn_captions: bool | None = None,
    width: int | None = None,
    height: int | None = None,
    fill: str | None = None,
    on_progress: ProgressFn | None = None,
) -> dict[str, Any]:
    """Cut and render one vertical clip from one or more spans.

    `spans` may be a single `(start, end)` pair or a list of them; adjacent
    spans are fused so a "stitch" that turns out contiguous stays a single cut.

    `fill` decides how a landscape frame meets the vertical canvas — "crop"
    fills it and keeps the subject in shot, "blur" letterboxes over a blurred
    copy, "auto" crops when a subject can be located.
    """
    width = width or config.RENDER_WIDTH
    height = height or config.RENDER_HEIGHT
    burn = config.BURN_CAPTIONS if burn_captions is None else burn_captions

    spans = timeline.merge_touching(timeline.as_spans(spans))
    if not spans:
        raise media.MediaError("render_clip called with no usable spans")

    out_path = Path(out_path)
    work_dir = out_path.parent
    work_dir.mkdir(parents=True, exist_ok=True)
    duration = max(0.1, timeline.total_duration(spans))

    # Sidecars describe the *clip* timeline, so they are rebased onto the stitch.
    ass_name = f"{out_path.stem}.ass"
    ass_path = work_dir / ass_name
    captions_mod.write_ass(
        ass_path,
        captions_mod.build_ass(segments, spans, width=width, height=height,
                               duration=duration),
    )
    srt_path = work_dir / f"{out_path.stem}.srt"
    srt_path.write_text(
        transcribe_mod.to_srt(timeline.segments_in_spans(segments, spans)),
        encoding="utf-8",
    )

    # --- how to fill the vertical frame ---
    fill_mode = (fill or config.RENDER_FILL or "crop").strip().lower()
    crop_box: dict[str, int] | None = None
    subject_x: float | None = None

    crop_tracks: list[list[dict[str, Any]]] = []

    if has_video and fill_mode in {"crop", "auto"}:
        probe = media.probe(source)
        src_w, src_h = int(probe.get("width") or 0), int(probe.get("height") or 0)
        if src_w and src_h:
            window = framing.window_fraction(src_w, src_h, width, height)
            if config.FRAMING_PER_SHOT:
                # One crop per shot rather than one per clip: an edit that cuts
                # to b-roll, or to the other person in the room, reframes on
                # the cut instead of keeping a compromise position all the way
                # through.
                words = boundaries.flatten_words(segments)
                crop_tracks = [
                    _subject_track(source, span, words, window=window,
                                   work_dir=work_dir)
                    for span in spans
                ]
                subject_x = framing.average_x(crop_tracks)
            else:
                subject_x = framing.find_subject_x(
                    source, spans, window=window,
                    samples=config.FRAMING_SAMPLES, work_dir=work_dir,
                )
            crop_box = framing.crop_geometry(src_w, src_h, width, height, subject_x)
            # Expressions need pixel positions, so resolve each shot's centre
            # against the same box the static path would have produced.
            if crop_box:
                for track in crop_tracks:
                    for entry in track:
                        shot_box = framing.crop_geometry(
                            src_w, src_h, width, height, entry["x"])
                        entry["x"] = (shot_box or crop_box)["x"]
            else:
                crop_tracks = []
        if fill_mode == "auto" and subject_x is None:
            fill_mode = "blur"
        elif fill_mode == "auto":
            fill_mode = "crop"
    elif fill_mode not in {"crop", "blur"}:
        fill_mode = "blur"

    backend = caption_backend(burn)
    subtitle_arg = ass_name if backend == "libass" else None
    overlays: list[dict[str, Any]] = []
    if backend == "pillow":
        overlays = pngcaptions.render_events(
            captions_mod.build_events(segments, spans, duration=duration),
            work_dir / f"{out_path.stem}-captions",
            width=width, height=height,
        )

    args: list[str] = [
        media.binary("ffmpeg"), "-y", "-hide_banner", "-loglevel", "error", "-nostats",
    ]

    color_input: int | None = None
    if not has_video:
        # Background canvas first, so the span inputs start at index 1.
        color_input = 0
        args += ["-f", "lavfi", "-t", f"{duration:.3f}",
                 "-i", f"color=c=0x0b0f14:s={width}x{height}:r=30"]

    span_offset = 0 if has_video else 1
    for start, end in spans:
        args += ["-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", str(source)]

    caption_offset = span_offset + len(spans)
    if overlays:
        args += pngcaptions.input_args(overlays)

    args += [
        "-filter_complex",
        _build_graph(
            spans, has_video=has_video, width=width, height=height,
            subtitle_file=subtitle_arg, overlays=overlays,
            color_input=color_input, span_offset=span_offset,
            caption_offset=caption_offset, fill_mode=fill_mode,
            crop_box=crop_box, crop_tracks=crop_tracks,
        ),
        "-map", "[vout]", "-map", "[aout]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-profile:v", "high", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2",
        "-movflags", "+faststart",
        "-progress", "pipe:1",
        str(out_path),
    ]

    try:
        _run_ffmpeg(args, duration=duration, cwd=work_dir, on_progress=on_progress)
    finally:
        pngcaptions.cleanup(overlays)

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise media.MediaError(f"render produced no output at {out_path}")

    poster = media.thumbnail(
        out_path, work_dir / f"{out_path.stem}.jpg", at=min(1.0, duration / 3), width=405
    )
    info = media.probe(out_path)

    return {
        "path": str(out_path),
        "filename": out_path.name,
        "thumbnail": str(poster) if poster else None,
        "subtitles_ass": str(ass_path),
        "subtitles_srt": str(srt_path),
        "spans": [{"start": s, "end": e} for s, e in spans],
        "stitched": len(spans) > 1,
        "fill": fill_mode,
        "crop": crop_box,
        "crop_track": [
            [{"start": round(e["start"], 2), "x": e["x"]} for e in track]
            for track in crop_tracks
        ] or None,
        "subject_x": None if subject_x is None else round(subject_x, 4),
        "captions_burned": backend != "none",
        "caption_backend": backend,
        "captions_skipped_reason": (
            None if backend != "none" or not burn
            else "no caption renderer available (ffmpeg lacks the `subtitles` "
                 "filter and Pillow is not installed)"
        ),
        "size_bytes": out_path.stat().st_size,
        "rendered_duration": round(info["duration"], 3),
        "width": info["width"],
        "height": info["height"],
    }
