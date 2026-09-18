"""Finding the parts of a video that must never become a clip.

Two kinds, both worthless as short-form and both prone to scoring *well*:
integrated sponsor reads, and channel housekeeping ("hit the subscribe
button"). Sponsor detection is the involved one and is described below;
housekeeping is a short marker match near the bottom of this module.

A baked-in ad is the worst possible clip: it scores well (it is scripted, it
has a hook, it is enthusiastic) and it is worthless. Detection leans on the
fact that sponsor reads announce themselves and then repeat a brand name:

    "shout out to Ridge for sponsoring this video"      <- seed
    "So Channel Sponsor Ridge has a bunch of accessories ..."
    "Ridge's PowerBank also works with ..."             <- brand mentions
    "So check out everything by heading over to Ridge.com/MKBHD."  <- CTA

So a strong marker seeds the region, the brand name is read out of that seed,
and the region grows over the neighbouring sentences that keep mentioning it.

Brand extraction is deliberately narrow — captured only from a template around
the marker — because a loose sweep for capitalised words picks up the video's
own subject ("iPhone") and would flag the entire video as an advert. A brand
appearing in a large share of the transcript is rejected for the same reason.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Sequence

from app.core import context

#: Phrases that on their own mean "this is an ad".
STRONG_MARKERS = (
    "sponsor", "sponsored", "sponsoring", "sponsorship",
    "brought to you by", "use code", "use my code", "promo code",
    "coupon code", "discount code", "link in the description",
    "link in the bio", "link below", "free trial", "affiliate link",
    "start your free", "sign up today", "today's video is brought",
)

#: Phrases that support a nearby strong marker but never fire alone — "check
#: out" and "thanks to" appear constantly in ordinary speech.
WEAK_MARKERS = (
    "check out", "check them out", "head over to", "heading over to",
    "head to", "go to", "sign up", "signing up", "discount", "% off",
    "percent off", "thanks to", "shout out to", "deal", "offer",
    "get yours", "available at", "learn more",
)

#: Brand captured from a template around the marker, not swept up loosely.
_BRAND_PATTERNS = (
    re.compile(r"sponsored\s+by\s+([A-Z][\w&'’.\-]*)", re.I | re.U),
    re.compile(r"brought\s+to\s+you\s+by\s+([A-Z][\w&'’.\-]*)", re.I | re.U),
    re.compile(r"(?:shout\s*out|thanks|thank\s+you)\s+to\s+([A-Z][\w&'’.\-]*)"
               r"\s+for\s+sponsor", re.I | re.U),
    re.compile(r"channel\s+sponsor\s+([A-Z][\w&'’.\-]*)", re.I | re.U),
    re.compile(r"(?:today's|todays|this\s+video's)\s+sponsor(?:\s+is)?\s+"
               r"([A-Z][\w&'’.\-]*)", re.I | re.U),
    re.compile(r"sponsor(?:ed|ing)?\s*(?:by|is)?[:,]?\s+([A-Z][\w&'’.\-]*)",
               re.I | re.U),
)

#: Whisper renders spoken URLs as "Ridge .com slash MKBHD".
_DOMAIN_RE = re.compile(r"\b([A-Za-z][\w\-]{2,})\s*\.\s*(?:com|co|io|net|org)\b")

#: Never treat these as a sponsor's name.
_BRAND_STOPLIST = {
    "the", "this", "that", "our", "their", "them", "today", "video", "channel",
    "sponsor", "sponsors", "watching", "everyone", "guys", "you", "and", "for",
    "all", "right", "so", "well", "also", "here", "there", "it", "its",
}

#: A sponsor read is time-bounded — a minute, rarely two. So a "brand" whose
#: mentions are scattered over more than this much of the video is the
#: programme's own subject ("iPhone"), not a sponsor.
#:
#: Measuring instead as a share of sentences does not work: on a short video a
#: perfectly legitimate read accounts for half of them, and the sponsor gets
#: thrown out along with the topic.
MAX_BRAND_SPREAD_SECONDS = 150.0
MAX_BRAND_SPREAD_FRACTION = 0.35


def _has(text: str, markers: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(m in lowered for m in markers)


def _brand_candidates(text: str) -> set[str]:
    """Brand names captured from sponsor templates in one sentence."""
    found: set[str] = set()
    for pattern in _BRAND_PATTERNS:
        for match in pattern.findall(text):
            name = match.strip(".,'’\"").lower()
            if len(name) >= 3 and name not in _BRAND_STOPLIST:
                found.add(name)
    for match in _DOMAIN_RE.findall(text):
        name = match.strip().lower()
        if len(name) >= 3 and name not in _BRAND_STOPLIST:
            found.add(name)
    return found


def _mentions(text: str, brand: str) -> bool:
    return re.search(rf"\b{re.escape(brand)}\b", text, re.I) is not None


#: A read's URL often lands a few sentences after the marker ("...sponsored.
#: Go to Squarespace.com for a discount"), so the window after a marker is
#: searched too rather than the marker's own sentence alone.
BRAND_LOOKAHEAD_SENTENCES = 8


def find_brands(sentences: Sequence[dict[str, Any]]) -> set[str]:
    """Sponsor names in the transcript, excluding the video's own subject."""
    candidates: set[str] = set()
    for i, sentence in enumerate(sentences):
        if not _has(sentence["text"], STRONG_MARKERS):
            continue
        window = sentences[i:i + 1 + BRAND_LOOKAHEAD_SENTENCES]
        for nearby in window:
            candidates |= _brand_candidates(nearby["text"])

    if not candidates or not sentences:
        return set()

    total = float(sentences[-1]["end"]) - float(sentences[0]["start"])
    budget = max(MAX_BRAND_SPREAD_SECONDS, MAX_BRAND_SPREAD_FRACTION * total)

    keep: set[str] = set()
    for brand in candidates:
        mentions = [s for s in sentences if _mentions(s["text"], brand)]
        if not mentions:
            continue
        spread = float(mentions[-1]["end"]) - float(mentions[0]["start"])
        if spread <= budget:
            keep.add(brand)
    return keep


def find_ad_spans(
    words: Sequence[dict[str, Any]],
    *,
    gap_seconds: float = 30.0,
    min_seconds: float = 6.0,
    pad_seconds: float = 0.0,
) -> list[dict[str, Any]]:
    """Sponsor regions in the transcript as `{start, end, brand, reason}`.

    A region is seeded by a strong marker and grown across nearby sentences
    that mention the sponsor's brand, so the whole read is covered rather than
    only the sentence that gave it away.
    """
    sentences = context.sentences(words)
    if not sentences:
        return []

    seeds = [i for i, s in enumerate(sentences) if _has(s["text"], STRONG_MARKERS)]
    if not seeds:
        return []

    brands = find_brands(sentences)

    members: dict[int, str] = {}
    for i, sentence in enumerate(sentences):
        if i in seeds:
            members[i] = "marker"
            continue
        hit = next((b for b in brands if _mentions(sentence["text"], b)), None)
        if hit:
            members[i] = f"brand:{hit}"

    if not members:
        return []

    # Cluster members that sit close together in time.
    ordered = sorted(members)
    clusters: list[list[int]] = [[ordered[0]]]
    for idx in ordered[1:]:
        previous = clusters[-1][-1]
        if sentences[idx]["start"] - sentences[previous]["end"] <= gap_seconds:
            clusters[-1].append(idx)
        else:
            clusters.append([idx])

    spans: list[dict[str, Any]] = []
    for cluster in clusters:
        # A cluster needs a marker in it; brand mentions alone are not enough,
        # or a passing reference to the sponsor elsewhere would be cut out.
        if not any(i in seeds for i in cluster):
            continue

        lo = min(cluster)
        hi = max(cluster)
        # A trailing call to action often lands just past the last brand
        # mention ("check out everything by heading over to ...").
        if hi + 1 < len(sentences) and _has(sentences[hi + 1]["text"], WEAK_MARKERS):
            hi += 1

        start = float(sentences[lo]["start"]) - pad_seconds
        end = float(sentences[hi]["end"]) + pad_seconds
        if end - start < min_seconds:
            continue
        brand = next(
            (members[i].split(":", 1)[1] for i in cluster
             if members[i].startswith("brand:")),
            None,
        )
        if brand is None:
            # A one-sentence read has no separate brand-mention member, so take
            # the name straight out of the marker sentence.
            named = sorted(
                b for i in cluster for b in _brand_candidates(sentences[i]["text"])
                if b in brands
            )
            brand = named[0] if named else None
        spans.append({
            "start": round(max(start, 0.0), 3),
            "end": round(end, 3),
            "brand": brand,
            "sentences": hi - lo + 1,
            "reason": f"sponsor read{f' for {brand}' if brand else ''}",
        })

    return spans


#: Channel housekeeping: not an advert, but equally worthless as a clip.
#: Kept narrow on purpose. "this channel" and "my channel" are *not* here —
#: "I've reviewed a lot of other foldables on this channel" is real content
#: that sits inside a perfectly good clip.
HOUSEKEEPING_MARKERS = (
    "subscribe", "subscribed", "hit the like", "like button", "smash that",
    "smash the", "notification bell", "notification squad", "ring the bell",
    "comment below", "in the comments below", "let me know in the comments",
    "merch", "patreon", "join this channel", "channel membership",
    "hit that bell", "turn on notifications",
)


#: Longest a housekeeping cut may run. These asks are brief; anything longer
#: means the marker was found inside a long passage of real content.
MAX_HOUSEKEEPING_SECONDS = 14.0

#: How far back from the marker to look for the start of the ask.
HOUSEKEEPING_LOOKBACK_SECONDS = 6.0


def find_housekeeping_spans(
    words: Sequence[dict[str, Any]],
    *,
    min_seconds: float = 0.5,
) -> list[dict[str, Any]]:
    """Subscribe asks, like prompts and similar channel admin.

    Located at *word* level rather than by sentence. Whisper regularly emits
    long unpunctuated runs — one real example is 37 seconds covering the phone
    colours and only ending with "...make sure you get subscribed" — so
    excluding the whole sentence would throw away half a minute of content to
    remove a five second ask.
    """
    from app.core import boundaries as _boundaries

    if not words:
        return []

    tokens = [(w, (w.get("word") or "").strip().lower()) for w in words]
    sentences = context.sentences(words)
    raw: list[tuple[float, float]] = []

    for i in range(len(tokens)):
        # Longest marker is ~4 words; join a short window and test each marker.
        window = " ".join(t for _w, t in tokens[i:i + 6])
        marker = next((m for m in HOUSEKEEPING_MARKERS if window.startswith(m)), None)
        if marker is None:
            continue

        start = float(tokens[i][0]["start"])
        # Walk back to the start of the clause the ask lives in, so the cut does
        # not begin mid-phrase.
        floor = start - HOUSEKEEPING_LOOKBACK_SECONDS
        for offset in range(i - 1, -1, -1):
            w = tokens[offset][0]
            if float(w["end"]) < floor:
                break
            # Punctuation is the best boundary, but these long unpunctuated
            # runs have none — so a breath counts as one too.
            gap = float(tokens[offset + 1][0]["start"]) - float(w["end"])
            if (_boundaries.ends_clause(w) or _boundaries.ends_sentence(w)
                    or gap >= 0.2):
                start = float(tokens[offset + 1][0]["start"])
                break
            start = float(w["start"])

        # Run to the end of the enclosing sentence: the ask finishes the thought.
        enclosing = next(
            (s for s in sentences
             if s["start"] <= float(tokens[i][0]["start"]) <= s["end"]),
            None,
        )
        end = float(enclosing["end"]) if enclosing else float(tokens[i][0]["end"])
        end = min(end, start + MAX_HOUSEKEEPING_SECONDS)
        if end - start >= min_seconds:
            raw.append((start, end))

    # Merge overlapping hits (a single ask often trips several markers).
    raw.sort()
    merged: list[list[float]] = []
    for start, end in raw:
        if merged and start <= merged[-1][1] + 0.25:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    return [
        {
            "start": round(start, 3),
            "end": round(end, 3),
            "kind": "housekeeping",
            "sentences": 1,
            "reason": "channel housekeeping",
        }
        for start, end in merged
    ]


def find_excluded_spans(
    words: Sequence[dict[str, Any]],
    *,
    gap_seconds: float = 30.0,
    min_seconds: float = 6.0,
    filler: bool = True,
) -> list[dict[str, Any]]:
    """Everything that must never become a clip, tagged by `kind`."""
    spans: list[dict[str, Any]] = []
    for span in find_ad_spans(words, gap_seconds=gap_seconds,
                              min_seconds=min_seconds):
        spans.append({**span, "kind": "sponsor"})
    if filler:
        spans += find_housekeeping_spans(words)
    spans.sort(key=lambda s: float(s["start"]))
    return spans


def subtract(
    span: tuple[float, float],
    excluded: Sequence[dict[str, Any]],
    *,
    min_piece: float = 2.0,
) -> list[tuple[float, float]]:
    """What is left of `span` once every excluded region is cut out of it.

    This is the excise-and-stitch primitive: a subscribe ask in the *middle* of
    an otherwise good clip should be removed and the halves joined, which
    `clamp_outside` cannot express — it truncates at the first excluded region
    and would reduce a 43s clip to a useless 8s head.
    """
    pieces: list[tuple[float, float]] = [(float(span[0]), float(span[1]))]
    for region in sorted(excluded, key=lambda r: float(r["start"])):
        lo, hi = float(region["start"]), float(region["end"])
        nxt: list[tuple[float, float]] = []
        for start, end in pieces:
            if hi <= start or lo >= end:        # no overlap
                nxt.append((start, end))
                continue
            if start < lo:
                nxt.append((start, min(lo, end)))
            if end > hi:
                nxt.append((max(hi, start), end))
        pieces = nxt

    return [
        (round(s, 3), round(e, 3)) for s, e in pieces if e - s >= min_piece
    ]


def overlap_seconds(span: tuple[float, float], other: tuple[float, float]) -> float:
    """Seconds shared by two spans."""
    return max(0.0, min(span[1], other[1]) - max(span[0], other[0]))


def overlap_fraction(
    span: tuple[float, float],
    ad_spans: Iterable[dict[str, Any]],
) -> float:
    """Fraction of `span` that falls inside any advert."""
    length = span[1] - span[0]
    if length <= 0:
        return 0.0
    covered = sum(
        overlap_seconds(span, (float(a["start"]), float(a["end"])))
        for a in ad_spans
    )
    return min(covered / length, 1.0)


def clamp_outside(
    span: tuple[float, float],
    ad_spans: Sequence[dict[str, Any]],
    *,
    min_seconds: float = 1.0,
) -> tuple[float, float] | None:
    """Shrink `span` so it stops short of any advert.

    Topic completion and boundary snapping both push a clip's end later, which
    can walk it straight into a sponsor read that started just afterwards.
    Returns None when nothing long enough survives.
    """
    start, end = span
    for ad in sorted(ad_spans, key=lambda a: float(a["start"])):
        ad_start, ad_end = float(ad["start"]), float(ad["end"])
        if overlap_seconds((start, end), (ad_start, ad_end)) <= 0:
            continue
        if start < ad_start:
            end = min(end, ad_start)          # stop before the advert
        elif end > ad_end:
            start = max(start, ad_end)        # start after it
        else:
            return None                       # wholly inside an advert
    if end - start < min_seconds:
        return None
    return (round(start, 3), round(end, 3))


def is_advert(
    span: tuple[float, float],
    ad_spans: Iterable[dict[str, Any]],
    *,
    threshold: float = 0.25,
) -> bool:
    """Whether enough of `span` is sponsored that it should not become a clip."""
    return overlap_fraction(span, list(ad_spans)) >= threshold
