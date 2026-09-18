"""Detecting integrated sponsor reads."""
from __future__ import annotations

from app.core import ads, context


def words_from(text: str, start: float = 0.0, step: float = 0.3) -> list[dict]:
    return [
        {"word": tok, "start": round(start + i * step, 3),
         "end": round(start + (i + 1) * step, 3)}
        for i, tok in enumerate(text.split())
    ]


# A shortened version of a real read: seed, brand body, CTA, then the outro.
SPONSORED = (
    "Buckle up and get subscribed. "
    "All right, and shout out to Ridge for sponsoring this video. "
    "So Channel Sponsor Ridge has a bunch of accessories for your phone. "
    "Their famous PowerBank offers five in one power delivery. "
    "Ridge's PowerBank also works with the Duo, wired or MagSafe. "
    "So check out everything by heading over to Ridge .com slash MKBHD. "
    "Thanks for watching and I'll catch you in the next one. "
)


# --- brand extraction ---

def test_brand_is_read_out_of_the_seed_sentence():
    sents = context.sentences(words_from(SPONSORED))
    assert "ridge" in ads.find_brands(sents)


def test_brand_extraction_ignores_the_videos_own_subject():
    """A loose sweep for capitalised words would flag the whole video.

    Written at realistic length: the subject recurs across the whole runtime
    while the sponsor read stays in one place, which is what separates them.
    """
    text = (
        "This video is sponsored by Ridge and their wallets are lovely. "
        + "The iPhone is great and here is another thought about phones. " * 90
    )
    sents = context.sentences(words_from(text))
    brands = ads.find_brands(sents)
    assert "ridge" in brands
    assert "iphone" not in brands, "the subject was mistaken for a sponsor"


def test_no_brands_without_a_sponsor_marker():
    sents = context.sentences(words_from(
        "The camera is good. Apple made it well. Check out the screen. "
    ))
    assert ads.find_brands(sents) == set()


def test_brand_captured_from_spoken_domain():
    sents = context.sentences(words_from(
        "This one is sponsored. Go to Squarespace .com slash me for a discount. "
    ))
    assert "squarespace" in ads.find_brands(sents)


# --- span detection ---

def test_finds_the_whole_read_not_just_the_seed():
    words = words_from(SPONSORED)
    spans = ads.find_ad_spans(words, gap_seconds=30.0, min_seconds=2.0)
    assert len(spans) == 1
    span = spans[0]
    assert span["brand"] == "ridge"
    # It must cover the body, not only the sentence containing "sponsoring".
    assert span["sentences"] >= 4


def test_outro_is_not_swallowed_by_the_ad():
    words = words_from(SPONSORED)
    sents = context.sentences(words)
    outro = next(s for s in sents if s["text"].startswith("Thanks for watching"))
    span = ads.find_ad_spans(words, min_seconds=2.0)[0]
    assert span["end"] <= outro["start"] + 0.01, "the outro was marked as an advert"


def test_intro_before_the_read_is_untouched():
    words = words_from(SPONSORED)
    sents = context.sentences(words)
    span = ads.find_ad_spans(words, min_seconds=2.0)[0]
    assert span["start"] >= sents[0]["end"], "content before the read was included"


def test_thanks_to_alone_is_not_an_advert():
    """Regression from real footage: '25% smaller thanks to moving the camera'."""
    words = words_from(
        "The dynamic island is 25 % smaller thanks to them moving the camera. "
        "It still houses Face ID though. "
    )
    assert ads.find_ad_spans(words) == []


def test_check_out_alone_is_not_an_advert():
    words = words_from(
        "Check out how thin this thing is. Go to the settings menu to see it. "
    )
    assert ads.find_ad_spans(words) == []


def test_no_markers_means_no_spans():
    assert ads.find_ad_spans(words_from("Just a normal video about phones. ")) == []
    assert ads.find_ad_spans([]) == []


def test_two_separate_reads_are_two_spans():
    text = (
        "This video is sponsored by Ridge and their wallets are nice. "
        + "Filler sentence about phones. " * 40
        + "Also sponsored by Squarespace, go to Squarespace .com today. "
    )
    spans = ads.find_ad_spans(words_from(text), gap_seconds=20.0, min_seconds=1.0)
    assert len(spans) == 2
    assert {s["brand"] for s in spans} == {"ridge", "squarespace"}


def test_a_stray_brand_mention_far_away_is_not_an_advert():
    text = (
        "This video is sponsored by Ridge, check out their wallets. "
        + "Filler about phones. " * 60
        + "I also happen to own a Ridge wallet myself, unrelated. "
    )
    spans = ads.find_ad_spans(words_from(text), gap_seconds=15.0, min_seconds=1.0)
    # The lone later mention has no marker, so it is not its own advert.
    assert len(spans) == 1


def test_min_seconds_rejects_a_trivial_hit():
    words = words_from("Sponsored. ")
    assert ads.find_ad_spans(words, min_seconds=30.0) == []


# --- overlap helpers ---

def test_overlap_fraction():
    adverts = [{"start": 100.0, "end": 200.0}]
    assert ads.overlap_fraction((0.0, 100.0), adverts) == 0.0
    assert ads.overlap_fraction((150.0, 250.0), adverts) == 0.5
    assert ads.overlap_fraction((120.0, 180.0), adverts) == 1.0
    assert ads.overlap_fraction((5.0, 5.0), adverts) == 0.0


def test_is_advert_threshold():
    adverts = [{"start": 100.0, "end": 200.0}]
    assert ads.is_advert((150.0, 250.0), adverts, threshold=0.25) is True
    assert ads.is_advert((190.0, 290.0), adverts, threshold=0.25) is False


def test_clamp_outside_trims_rather_than_dropping():
    adverts = [{"start": 100.0, "end": 160.0}]
    assert ads.clamp_outside((80.0, 120.0), adverts) == (80.0, 100.0)
    assert ads.clamp_outside((150.0, 200.0), adverts) == (160.0, 200.0)
    assert ads.clamp_outside((10.0, 60.0), adverts) == (10.0, 60.0)


def test_clamp_outside_drops_a_span_inside_an_advert():
    adverts = [{"start": 100.0, "end": 160.0}]
    assert ads.clamp_outside((110.0, 150.0), adverts) is None


def test_clamp_outside_drops_what_is_left_too_short():
    adverts = [{"start": 100.0, "end": 160.0}]
    assert ads.clamp_outside((99.5, 140.0), adverts, min_seconds=1.0) is None


def test_clamp_outside_handles_several_adverts():
    adverts = [{"start": 100.0, "end": 130.0}, {"start": 200.0, "end": 230.0}]
    assert ads.clamp_outside((50.0, 90.0), adverts) == (50.0, 90.0)
    assert ads.clamp_outside((140.0, 190.0), adverts) == (140.0, 190.0)


# --- channel housekeeping ---

def test_subscribe_ask_is_found():
    words = words_from(
        "The colours are lovely this year. "
        "So just hit the subscribe button down there, just click it, okay great. "
        "Now the pro phones start at 1199. "
    )
    spans = ads.find_housekeeping_spans(words)
    assert len(spans) == 1
    assert spans[0]["kind"] == "housekeeping"


def test_housekeeping_does_not_swallow_a_long_content_run():
    """Whisper emits long unpunctuated runs; only the ask may be cut.

    Real example: 37 seconds about phone colours ending in "...make sure you
    get subscribed". Cutting the whole sentence would lose half a minute.
    """
    long_run = (
        "And then yeah pretty much the only other new thing here is the colours "
        "so we can enjoy the new burgundy which seems to be the flagship colour "
        "and there is also glacier which is a light blue and a new silver which "
        "is kind of boring but more uniform than last year and finally a fully "
        "matte black which I think is the one that is really cool and I think "
        "the logical thing to do right now is just to make sure you get subscribed."
    )
    words = words_from(long_run)
    spans = ads.find_housekeeping_spans(words)
    assert len(spans) == 1
    cut = spans[0]["end"] - spans[0]["start"]
    assert cut <= ads.MAX_HOUSEKEEPING_SECONDS + 0.01
    # The colour discussion at the start must survive.
    assert spans[0]["start"] > words[10]["end"], "cut reaches back into the content"


def test_channel_mention_is_not_housekeeping():
    """Regression: "I've reviewed a lot of other foldables on this channel"
    is real content sitting inside a good clip."""
    words = words_from(
        "But also I've reviewed a lot of other foldables on this channel. "
        "You might have watched a bunch of those other videos. "
    )
    assert ads.find_housekeeping_spans(words) == []


def test_housekeeping_merges_adjacent_asks():
    words = words_from(
        "Make sure you get subscribed so you see the reviews. "
        "And hit the like button while you are there. "
    )
    spans = ads.find_housekeeping_spans(words)
    assert len(spans) == 1, "two adjacent asks should be one cut"


def test_no_housekeeping_in_ordinary_speech():
    assert ads.find_housekeeping_spans(words_from(
        "The battery lasts all day and the screen is bright. "
    )) == []


def test_find_excluded_spans_tags_both_kinds():
    words = words_from(
        "This video is sponsored by Ridge and their wallets are great. "
        "Go to Ridge .com slash me for a discount today. "
        + "Filler about phones. " * 30
        + "Also please hit the subscribe button before you go. "
    )
    spans = ads.find_excluded_spans(words, gap_seconds=20.0, min_seconds=1.0)
    kinds = {s["kind"] for s in spans}
    assert kinds == {"sponsor", "housekeeping"}
    assert spans == sorted(spans, key=lambda s: s["start"])


def test_find_excluded_spans_can_skip_filler():
    words = words_from(
        "This video is sponsored by Ridge, go to Ridge .com for a discount. "
        "Also hit the subscribe button. "
    )
    spans = ads.find_excluded_spans(words, gap_seconds=20.0, min_seconds=1.0,
                                    filler=False)
    assert all(s["kind"] == "sponsor" for s in spans)


# --- subtract: excise and stitch ---

def test_subtract_keeps_both_sides_of_interior_filler():
    """The whole point: a subscribe ask mid-clip must not truncate the clip.

    `clamp_outside` would stop at the filler and return only the short head.
    """
    excluded = [{"start": 203.0, "end": 209.5}]
    pieces = ads.subtract((194.9, 237.9), excluded, min_piece=2.0)
    assert len(pieces) == 2
    assert pieces[0] == (194.9, 203.0)
    assert pieces[1] == (209.5, 237.9)


def test_subtract_drops_pieces_below_the_minimum():
    excluded = [{"start": 196.0, "end": 209.5}]
    pieces = ads.subtract((194.9, 237.9), excluded, min_piece=2.0)
    assert pieces == [(209.5, 237.9)], "the 1.1s head should be discarded"


def test_subtract_handles_filler_at_each_edge():
    excluded = [{"start": 0.0, "end": 10.0}, {"start": 50.0, "end": 60.0}]
    assert ads.subtract((5.0, 55.0), excluded, min_piece=2.0) == [(10.0, 50.0)]


def test_subtract_returns_nothing_when_wholly_excluded():
    excluded = [{"start": 0.0, "end": 100.0}]
    assert ads.subtract((10.0, 50.0), excluded, min_piece=2.0) == []


def test_subtract_is_a_no_op_when_clear():
    span = (10.0, 50.0)
    assert ads.subtract(span, [{"start": 200.0, "end": 300.0}]) == [span]
    assert ads.subtract(span, []) == [span]


def test_subtract_handles_several_interior_regions():
    excluded = [{"start": 20.0, "end": 25.0}, {"start": 35.0, "end": 40.0}]
    pieces = ads.subtract((10.0, 50.0), excluded, min_piece=2.0)
    assert pieces == [(10.0, 20.0), (25.0, 35.0), (40.0, 50.0)]
