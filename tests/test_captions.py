"""Word-timed ASS caption generation."""
from __future__ import annotations

from app.core import captions


def test_collect_words_rebases_times_to_clip_start(transcript):
    words = captions.collect_words(transcript, [(10.0, 25.0)])
    assert words
    assert min(w["start"] for w in words) >= 0.0
    assert max(w["end"] for w in words) <= 15.0 + 0.01
    assert all(w["end"] >= w["start"] for w in words)


def test_collect_words_excludes_out_of_range(transcript):
    assert captions.collect_words(transcript, [(10_000.0, 10_010.0)]) == []


def test_collect_words_skips_blank_tokens():
    segs = [{
        "start": 0.0, "end": 2.0, "text": "hi",
        "words": [
            {"word": "  ", "start": 0.0, "end": 0.5},
            {"word": "hello", "start": 0.5, "end": 1.0},
        ],
    }]
    words = captions.collect_words(segs, [(0.0, 2.0)])
    assert [w["word"] for w in words] == ["hello"]


def test_build_ass_has_header_and_events(transcript):
    doc = captions.build_ass(transcript, [(5.0, 30.0)], width=1080, height=1920)
    assert "[Script Info]" in doc
    assert "PlayResX: 1080" in doc
    assert "PlayResY: 1920" in doc
    assert "[V4+ Styles]" in doc
    assert doc.count("Dialogue:") > 0


def test_build_ass_events_stay_inside_clip_duration(transcript):
    start, end = 5.0, 25.0
    doc = captions.build_ass(transcript, [(start, end)])

    def to_seconds(stamp: str) -> float:
        h, m, rest = stamp.split(":")
        s, cs = rest.split(".")
        return int(h) * 3600 + int(m) * 60 + int(s) + int(cs) / 100

    seen = 0
    for line in doc.splitlines():
        if not line.startswith("Dialogue:"):
            continue
        seen += 1
        fields = line.split(",")
        ev_start, ev_end = to_seconds(fields[1]), to_seconds(fields[2])
        assert 0.0 <= ev_start <= (end - start) + 0.01
        assert ev_start < ev_end <= (end - start) + 0.01
    assert seen > 0


def test_build_ass_without_words_returns_header_only():
    segs = [{"start": 0.0, "end": 5.0, "text": "no word timings", "words": []}]
    doc = captions.build_ass(segs, [(0.0, 5.0)])
    assert "[Events]" in doc
    assert "Dialogue:" not in doc


def test_build_ass_highlights_one_word_per_event(transcript):
    doc = captions.build_ass(transcript, [(0.0, 20.0)])
    dialogues = [l for l in doc.splitlines() if l.startswith("Dialogue:")]
    assert dialogues
    for line in dialogues:
        # Exactly one word is recoloured to the active colour per event.
        assert line.count(captions.ACTIVE_COLOUR) == 1


def test_escape_neutralises_ass_override_syntax():
    assert captions._escape("{\\an8}hi") == "\\{\\\\an8\\}hi"
    assert "\n" not in captions._escape("a\nb")


def test_timestamp_formatting():
    assert captions._ts(0) == "0:00:00.00"
    assert captions._ts(1.5) == "0:00:01.50"
    assert captions._ts(61.25) == "0:01:01.25"
    assert captions._ts(3661.5) == "1:01:01.50"
    assert captions._ts(-5) == "0:00:00.00"


def test_write_ass_creates_parent_dirs(tmp_path):
    target = tmp_path / "nested" / "deeper" / "clip.ass"
    captions.write_ass(target, "content")
    assert target.read_text() == "content"


def test_groups_respect_word_and_char_caps(transcript):
    words = captions.collect_words(transcript, [(0.0, 40.0)])
    for group in captions._group(words):
        assert 1 <= len(group) <= captions.MAX_WORDS_PER_GROUP


# --- stitched clips ---

def test_collect_words_rebases_across_two_spans(transcript):
    spans = [(0.0, 8.0), (40.0, 48.0)]
    words = captions.collect_words(transcript, spans)
    assert words
    # Content from the second span must sit right after the first, not at 40s.
    assert max(w["end"] for w in words) <= 16.01
    assert [w["start"] for w in words] == sorted(w["start"] for w in words)


def test_build_ass_covers_both_spans_of_a_stitch(transcript):
    spans = [(0.0, 8.0), (40.0, 48.0)]
    doc = captions.build_ass(transcript, spans)
    dialogues = [l for l in doc.splitlines() if l.startswith("Dialogue:")]
    assert dialogues

    def to_seconds(stamp):
        h, m, rest = stamp.split(":")
        s, cs = rest.split(".")
        return int(h) * 3600 + int(m) * 60 + int(s) + int(cs) / 100

    ends = [to_seconds(l.split(",")[2]) for l in dialogues]
    assert max(ends) <= 16.01
    # Captions should exist on both sides of the stitch point at 8s.
    assert any(e <= 8.0 for e in ends) and any(e > 8.0 for e in ends)


def test_build_events_marks_one_active_word_each(transcript):
    events = captions.build_events(transcript, [(0.0, 20.0)])
    assert events
    for event in events:
        assert 0 <= event.active < len(event.tokens)
        assert event.end > event.start
        assert event.tokens


def test_build_events_empty_without_speech():
    segs = [{"start": 0.0, "end": 5.0, "text": "x", "words": []}]
    assert captions.build_events(segs, [(0.0, 5.0)]) == []


def test_caption_events_never_overlap(transcript):
    """Two simultaneous events render as two stacked lines of subtitles.

    Regression from real footage: events took their start from the previous
    word's end but their end from the next word's start, so every inter-word
    silence produced an overlapping pair.
    """
    events = captions.build_events(transcript, [(0.0, 40.0)])
    assert events
    for a, b in zip(events, events[1:]):
        assert b.start >= a.end - 1e-6, (
            f"events overlap: {a.start}-{a.end} then {b.start}-{b.end}"
        )


def test_caption_events_overlap_free_with_gappy_word_timings():
    """Wide gaps between words are exactly what triggered the overlap."""
    segs = [{
        "start": 0.0, "end": 8.0, "text": "one two three four",
        "words": [
            {"word": "one", "start": 0.0, "end": 0.5},
            {"word": "two", "start": 1.5, "end": 2.0},     # 1.0s gap
            {"word": "three", "start": 3.5, "end": 4.0},   # 1.5s gap
            {"word": "four", "start": 6.0, "end": 6.5},    # 2.0s gap
        ],
    }]
    events = captions.build_events(segs, [(0.0, 8.0)])
    assert len(events) == 4
    for a, b in zip(events, events[1:]):
        assert b.start >= a.end - 1e-6


def test_caption_events_tile_without_gaps(transcript):
    """Tiling matters too: a gap between events makes the phrase flicker."""
    events = captions.build_events(transcript, [(0.0, 20.0)])
    same_phrase = [
        (a, b) for a, b in zip(events, events[1:]) if a.tokens == b.tokens
    ]
    assert same_phrase
    for a, b in same_phrase:
        assert abs(b.start - a.end) < 1e-6, "flicker between words of one phrase"


def test_ass_document_has_no_overlapping_dialogue(transcript):
    doc = captions.build_ass(transcript, [(0.0, 40.0)])

    def secs(stamp):
        h, m, rest = stamp.split(":")
        s, cs = rest.split(".")
        return int(h) * 3600 + int(m) * 60 + int(s) + int(cs) / 100

    spans = []
    for line in doc.splitlines():
        if line.startswith("Dialogue:"):
            f = line.split(",")
            spans.append((secs(f[1]), secs(f[2])))
    assert spans
    for (a_start, a_end), (b_start, b_end) in zip(spans, spans[1:]):
        assert b_start >= a_end - 0.011, \
            f"ASS events overlap: {a_start}-{a_end} vs {b_start}-{b_end}"


def test_fade_shrinks_for_very_short_events():
    """A fixed fade would leave a fast-speech caption washed out."""
    segs = [{
        "start": 0.0, "end": 2.0, "text": "one two three four five six",
        "words": [
            {"word": w, "start": round(i * 0.14, 3), "end": round((i + 1) * 0.14, 3)}
            for i, w in enumerate("one two three four five six".split())
        ],
    }]
    doc = captions.build_ass(segs, [(0.0, 2.0)])
    fades = [
        int(part.split("(")[1].split(",")[0])
        for part in doc.split("\\fad")[1:]
    ]
    assert fades, "no fade tags emitted"
    assert max(fades) < 60, f"fade not scaled down for short events: {fades}"
    assert min(fades) >= 10


def test_fade_stays_full_for_comfortable_events(transcript):
    doc = captions.build_ass(transcript, [(0.0, 40.0)])
    fades = [
        int(part.split("(")[1].split(",")[0])
        for part in doc.split("\\fad")[1:]
    ]
    assert max(fades) == 60, "long events should keep the full fade"
