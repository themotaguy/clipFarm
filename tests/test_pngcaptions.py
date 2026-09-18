"""The libass-free caption renderer."""
from __future__ import annotations

import pytest

from app.core import pngcaptions
from app.core.captions import CaptionEvent

pytestmark = pytest.mark.skipif(not pngcaptions.available(),
                                reason="Pillow not installed")


def events(n: int = 3) -> list[CaptionEvent]:
    return [
        CaptionEvent(start=i * 1.0, end=(i + 1) * 1.0,
                     tokens=["paying", "for", "a", "sports"], active=i % 4)
        for i in range(n)
    ]


def test_finds_a_usable_font():
    assert pngcaptions.find_font_file() is not None


def test_renders_one_png_per_event(tmp_path):
    out = pngcaptions.render_events(events(3), tmp_path, width=1080, height=1920)
    assert len(out) == 3
    for overlay in out:
        assert overlay["width"] > 0 and overlay["height"] > 0
        assert overlay["end"] > overlay["start"]


def test_glyphs_fit_inside_the_canvas(tmp_path):
    """Descenders must not be clipped: the canvas is sized from font metrics."""
    from PIL import Image

    # 'pygj' are all descenders; 'A' sets the ascender.
    ev = [CaptionEvent(start=0.0, end=1.0, tokens=["Apply", "juggling"], active=0)]
    out = pngcaptions.render_events(ev, tmp_path, width=1080, height=1920)
    image = Image.open(out[0]["path"]).convert("RGBA")
    alpha = image.getchannel("A")
    bbox = alpha.getbbox()
    assert bbox is not None, "nothing was drawn"
    left, top, right, bottom = bbox
    # Ink must not touch the canvas edges, or something was cut off.
    assert top > 0, "text touches the top edge"
    assert bottom < image.height, "text touches the bottom edge (descender clipped)"
    assert left > 0 and right < image.width


def test_active_word_is_drawn_in_the_highlight_colour(tmp_path):
    from PIL import Image

    ev = [CaptionEvent(start=0.0, end=1.0, tokens=["alpha", "beta"], active=1)]
    out = pngcaptions.render_events(ev, tmp_path, width=1080, height=1920)
    colours = {c[1] for c in Image.open(out[0]["path"]).convert("RGB").getcolors(1 << 20)}
    assert any(abs(r - 255) < 12 and abs(g - 227) < 14 and abs(b - 71) < 20
               for r, g, b in colours), "highlight colour missing"


def test_long_phrases_wrap_rather_than_overflow(tmp_path):
    ev = [CaptionEvent(start=0.0, end=1.0,
                       tokens=["absolutely", "enormous", "wordage", "everywhere"],
                       active=0)]
    out = pngcaptions.render_events(ev, tmp_path, width=1080, height=1920)
    assert out[0]["width"] <= 1080, "caption is wider than the frame"


def test_collapses_when_there_are_too_many_events(tmp_path, monkeypatch):
    monkeypatch.setattr(pngcaptions, "MAX_OVERLAYS", 4)
    out = pngcaptions.render_events(events(12), tmp_path, width=540, height=960)
    # 12 word-events over 3 distinct phrases collapse to far fewer images.
    assert len(out) < 12


def test_empty_events_produce_no_overlays(tmp_path):
    assert pngcaptions.render_events([], tmp_path, width=540, height=960) == []


def test_input_args_loop_each_image():
    overlays = [{"path": "/tmp/a.png", "start": 0.0, "end": 2.0,
                 "width": 10, "height": 10}]
    args = pngcaptions.input_args(overlays)
    assert args[:2] == ["-loop", "1"]
    assert "/tmp/a.png" in args


def test_filter_chain_enables_each_overlay_for_its_own_window():
    overlays = [
        {"path": "a.png", "start": 0.0, "end": 1.5, "width": 10, "height": 100},
        {"path": "b.png", "start": 1.5, "end": 3.0, "width": 10, "height": 100},
    ]
    chain = pngcaptions.filter_chain(
        overlays, first_input=2, in_label="vcat", out_label="vout", height=1920
    )
    assert "between(t,0.000,1.500)" in chain
    assert "between(t,1.500,3.000)" in chain
    assert chain.startswith("[vcat][2:v]overlay")
    assert chain.endswith("[vout]")


def test_filter_chain_is_a_passthrough_without_overlays():
    chain = pngcaptions.filter_chain(
        [], first_input=1, in_label="vcat", out_label="vout", height=1920
    )
    assert chain == "[vcat]null[vout]"


def test_cleanup_removes_the_image_directory(tmp_path):
    out = pngcaptions.render_events(events(2), tmp_path / "caps",
                                    width=540, height=960)
    assert (tmp_path / "caps").exists()
    pngcaptions.cleanup(out)
    assert not (tmp_path / "caps").exists()
