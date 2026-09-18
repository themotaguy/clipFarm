"""Transcript helpers — SRT rendering and text joining (no Whisper involved)."""
from __future__ import annotations

from app.core import transcribe


def test_srt_time_formatting():
    assert transcribe._srt_time(0) == "00:00:00,000"
    assert transcribe._srt_time(1.5) == "00:00:01,500"
    assert transcribe._srt_time(61.25) == "00:01:01,250"
    assert transcribe._srt_time(3661.007) == "01:01:01,007"


def test_to_srt_numbers_entries_from_one(transcript):
    out = transcribe.to_srt(transcript[:3])
    lines = out.splitlines()
    assert lines[0] == "1"
    assert "-->" in lines[1]
    assert out.count("-->") == 3


def test_to_srt_offset_rebases_to_clip_start(transcript):
    segs = transcript[2:5]
    offset = segs[0]["start"]
    out = transcribe.to_srt(segs, offset=offset)
    assert out.splitlines()[1].startswith("00:00:00,000")


def test_to_srt_never_emits_negative_times(transcript):
    out = transcribe.to_srt(transcript[:2], offset=10_000.0)
    assert "-" not in out.split("-->")[0].splitlines()[-1]
    assert out.count("00:00:00,000") >= 2


def test_to_srt_empty_input():
    assert transcribe.to_srt([]) == ""


def test_full_text_joins_segments(transcript):
    text = transcribe.full_text(transcript)
    assert text.startswith("Here is the thing")
    assert text.endswith("next time.")
    assert "  " not in text


def test_full_text_empty():
    assert transcribe.full_text([]) == ""


def test_hms_formatting():
    assert transcribe._hms(45) == "0:45"
    assert transcribe._hms(125) == "2:05"
    assert transcribe._hms(3725) == "1:02:05"
    assert transcribe._hms(-3) == "0:00"
