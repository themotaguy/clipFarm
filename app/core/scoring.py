"""Stages 4 & 5 — semantic retrieval of candidates, then LLM virality scoring.

Retrieval runs a fixed panel of "what tends to travel" probe queries against the
job's Chroma collection. Whatever survives that filter is handed to a local
Ollama model, which grades it and picks tighter in/out points.
"""

from __future__ import annotations

import json
import re
import threading
from typing import Any, Sequence

import config
from app.core import ads, boundaries, context, narrative, timeline, topics
from app.core.chunker import overlap_ratio

# Probe queries. Each targets a different reason a clip gets shared, so a chunk
# that lights up several probes is a broader hit than one that matches only one.
VIRALITY_PROBES = [
    "a surprising or counterintuitive claim that contradicts what most people believe",
    "an emotional, vulnerable, or deeply personal story",
    "concrete actionable advice, a practical tip, or a step-by-step method",
    "a bold, controversial, or contrarian opinion stated with conviction",
    "a shocking statistic, number, study result, or hard evidence",
    "a funny joke, witty comeback, or genuinely humorous moment",
    "a memorable quotable one-liner or piece of hard-won wisdom",
    "a dramatic turning point, revelation, or the answer to a big question",
]

SYSTEM_PROMPT = """\
You are a short-form video editor who has produced thousands of clips that went \
viral on TikTok, Reels and YouTube Shorts. You judge transcript excerpts for \
their potential as standalone vertical clips.

You are strict. Most excerpts are mediocre: rambling, context-dependent, or \
lacking a hook. Score those in the 20-50 range. Reserve 80+ for excerpts that \
genuinely stop a scroll.

Reply with a single JSON object and nothing else."""

USER_TEMPLATE = """\
Below is a numbered excerpt from a long-form video transcript. Each line is one \
spoken sentence with its timestamp.

--- EXCERPT ---
{lines}
--- END EXCERPT ---

Judge this excerpt as a candidate standalone short-form clip.

Also choose the tightest in/out points. The clip must be between {min_s:.0f} and \
{max_s:.0f} seconds long, and:

* START on the hook. "start_line" must be the same line as "hook_line" unless \
the hook genuinely needs one line of run-up. Never open on a request to \
subscribe or like, on a price or spec recitation, or on admin talk.
* END after the payoff. If the excerpt builds towards a reveal — "the one more \
thing", "wait until you see" — the clip must include the reveal itself, not \
stop on the promise of it.

Lines marked with `>` are the moment to build the clip around. Unmarked lines \
are surrounding context, shown only so you can pick setup (below) from them.

Setup: check whether the clip's opening refers to something it never explains — \
a phrase like "the third move", "that number", "this approach", "as I said", or \
any pronoun with no antecedent inside the clip. If it does, choose the shortest \
earlier passage that supplies the missing piece and return it as \
"setup_start_line" and "setup_end_line" (under {setup_s:.0f} seconds). It will be \
cut and shown immediately before the main clip. If the clip already explains \
itself, return null for both.

Return exactly this JSON shape:
{{
  "title": "punchy clip title, max 60 characters, no quotes around it",
  "setup_start_line": <integer line number or null>,
  "setup_end_line": <integer line number or null>,
  "hook": "the single most scroll-stopping sentence from the excerpt, verbatim",
  "hook_line": <integer line number the hook is quoted from>,
  "summary": "one sentence describing what the clip is about",
  "hook_strength": 0-10,
  "emotional_impact": 0-10,
  "standalone_clarity": 0-10,
  "shareability": 0-10,
  "virality_score": 0-100,
  "start_line": <integer line number>,
  "end_line": <integer line number>,
  "tags": ["two", "to", "four", "lowercase", "topic", "tags"],
  "reason": "one sentence on why this would or would not travel"
}}"""

_llm = None
_llm_lock = threading.Lock()


def get_llm():
    """Cached ChatOllama handle pinned to JSON output mode."""
    global _llm
    with _llm_lock:
        if _llm is None:
            from langchain_ollama import ChatOllama

            _llm = ChatOllama(
                model=config.OLLAMA_MODEL,
                base_url=config.OLLAMA_BASE_URL,
                temperature=0.3,
                format="json",
                num_ctx=config.OLLAMA_NUM_CTX,
            )
    return _llm


# --- retrieval ---

def retrieve_candidates(
    index: Any,
    limit: int | None = None,
    per_probe: int = 6,
) -> list[dict[str, Any]]:
    """Rank chunks by probe-query relevance and return the top `limit`."""
    limit = limit or config.SCORE_CANDIDATES
    merged = index.multi_query(VIRALITY_PROBES, k=per_probe)
    if not merged:
        return []

    candidates: list[dict[str, Any]] = []
    for hit in merged.values():
        meta = hit["metadata"]
        # Breadth bonus: matching several distinct probes beats one lucky match.
        breadth = min(hit["match_count"] / max(len(VIRALITY_PROBES) / 2, 1), 1.0)
        candidates.append({
            "chunk_id": hit["id"],
            "text": hit["text"],
            "start": float(meta.get("start", 0.0)),
            "end": float(meta.get("end", 0.0)),
            "relevance": hit["relevance"],
            "match_count": hit["match_count"],
            "matched_queries": [m["query"] for m in hit["matched_queries"]],
            "retrieval_score": round(0.75 * hit["relevance"] + 0.25 * breadth, 4),
        })

    candidates.sort(key=lambda c: c["retrieval_score"], reverse=True)
    # Thin out near-duplicate spans before spending LLM time on them.
    return dedupe_spans(candidates, max_overlap=0.6)[:limit]


def duration_buckets(
    count: int,
    min_seconds: float | None = None,
    max_seconds: float | None = None,
) -> list[tuple[float, float]]:
    """Split the allowed clip length into `count` ranges, shortest first.

    Spacing is geometric rather than linear: the difference between a 15s and a
    30s clip matters far more than between 150s and 165s, so the short end gets
    finer buckets.
    """
    lo = float(min_seconds if min_seconds is not None else config.CLIP_MIN_SECONDS)
    hi = float(max_seconds if max_seconds is not None else config.CLIP_MAX_SECONDS)
    count = max(1, int(count))
    if hi <= lo or count == 1:
        return [(lo, hi)]

    ratio = (hi / lo) ** (1.0 / count)
    edges = [lo * ratio ** i for i in range(count + 1)]
    edges[-1] = hi
    return [(round(edges[i], 2), round(edges[i + 1], 2)) for i in range(count)]


def spread_by_duration(
    clips: Sequence[dict[str, Any]],
    count: int,
    *,
    min_seconds: float | None = None,
    max_seconds: float | None = None,
) -> list[dict[str, Any]]:
    """Pick `count` clips whose lengths are spread across the allowed range.

    Clips arrive sorted by score. Taking the top N tends to return several of
    roughly the same length, so each duration bucket claims its best-scoring
    clip first; any unfilled buckets then fall back to score order.
    """
    buckets = duration_buckets(count, min_seconds, max_seconds)
    chosen: list[dict[str, Any]] = []
    used: set[int] = set()

    for lo, hi in buckets:
        for i, clip in enumerate(clips):
            if i in used:
                continue
            duration = float(clip.get("duration") or 0.0)
            # The last bucket owns anything above its floor.
            if lo <= duration <= hi or (hi >= buckets[-1][1] and duration > hi):
                chosen.append(clip)
                used.add(i)
                break

    # Backfill on score wherever a bucket found nothing.
    for i, clip in enumerate(clips):
        if len(chosen) >= count:
            break
        if i not in used:
            chosen.append(clip)
            used.add(i)

    chosen.sort(key=lambda c: c.get("final_score", 0.0), reverse=True)
    return chosen[:count]


def dedupe_spans(
    items: Sequence[dict[str, Any]],
    max_overlap: float = 0.5,
) -> list[dict[str, Any]]:
    """Greedy non-maximum suppression over time spans (input must be pre-sorted)."""
    kept: list[dict[str, Any]] = []
    for item in items:
        span = (item["start"], item["end"])
        if any(overlap_ratio(span, (k["start"], k["end"])) > max_overlap for k in kept):
            continue
        kept.append(item)
    return kept


# --- LLM scoring ---

def build_lines(
    segments: Sequence[dict[str, Any]],
    start: float,
    end: float,
) -> list[dict[str, Any]]:
    """Transcript lines overlapping [start, end], numbered for the prompt."""
    lines: list[dict[str, Any]] = []
    for seg in segments:
        if seg["end"] <= start or seg["start"] >= end:
            continue
        lines.append({
            "n": len(lines),
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"],
        })
    return lines


def _format_lines(
    lines: Sequence[dict[str, Any]],
    highlight: Sequence[int] = (),
) -> str:
    """Number the lines, marking the retrieved moment with `>`.

    The excerpt is padded with surrounding context so the model has something
    to draw setup from; the marker keeps it clear which part was retrieved.
    """
    marked = set(highlight)
    return "\n".join(
        f"{'>' if l['n'] in marked else ' '}[{l['n']}] "
        f"({l['start']:.1f}s–{l['end']:.1f}s) {l['text']}"
        for l in lines
    )


def score_candidate(
    candidate: dict[str, Any],
    lines: Sequence[dict[str, Any]],
    *,
    words: Sequence[dict[str, Any]] = (),
    media_duration: float | None = None,
    topic_boundaries: Sequence[float] = (),
    duration_range: tuple[float, float] | None = None,
    ad_spans: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    """Grade one candidate with Ollama and refine its in/out points.

    `words` are the transcript's word timings; given them, cut points are moved
    onto real pauses instead of landing on a word edge.
    """
    if not lines:
        return _fallback(candidate, "no transcript lines in range",
                         words=words, media_duration=media_duration)

    payoff = [
        l["n"] for l in lines
        if l["end"] > candidate["start"] and l["start"] < candidate["end"]
    ] or [l["n"] for l in lines]

    prompt = USER_TEMPLATE.format(
        lines=_format_lines(lines, payoff),
        min_s=config.CLIP_MIN_SECONDS,
        max_s=config.CLIP_MAX_SECONDS,
        setup_s=config.SETUP_MAX_SECONDS,
    )

    raw = _invoke(prompt)
    data = _parse_json(raw)
    if data is None:
        # One retry with a blunter instruction before giving up on the LLM.
        raw = _invoke(prompt + "\n\nOutput ONLY the JSON object. No prose, no markdown.")
        data = _parse_json(raw)
    if data is None:
        return _fallback(candidate, "model did not return parseable JSON",
                         words=words, media_duration=media_duration)

    main = _refine_bounds(lines, data, candidate, (payoff[0], payoff[-1]),
                          duration_range)

    # The scorer names the hook but keeps starting the clip well before it, and
    # in the worst cases ends *just* before it. Re-anchor onto it.
    hook_line, hook_score = narrative.locate_hook(
        _as_text(data.get("hook")), data.get("hook_line"), lines,
        threshold=config.HOOK_MATCH_THRESHOLD,
    )
    hook_applied = False
    if config.HOOK_ANCHOR and hook_line is not None:
        # How far we may cut to reach the hook depends on how sure we are of
        # where it is. A near-exact quote can be trusted to skip a long
        # preamble; a fuzzy match gets the conservative cap so a mislocated
        # hook cannot gut the clip.
        max_trim = (
            config.HOOK_TRIM_STRONG_SECONDS
            if hook_score >= config.HOOK_STRONG_MATCH
            else config.HOOK_TRIM_MAX_SECONDS
        )
        anchored = narrative.anchor_to_hook(
            main, hook_line,
            run_up=config.HOOK_RUN_UP_SECONDS,
            max_trim=max_trim,
            ceiling=_ceiling(duration_range),
        )
        hook_applied = anchored != main
        main = anchored

    main = _finish_thought(main, topic_boundaries, duration_range, ad_spans)

    # Never leave a buildup hanging: if the payoff starts just after the cut,
    # take it in.
    reveal_applied = False
    if config.REVEAL_COMPLETION and words:
        sentences = context.sentences(words)
        completed = narrative.complete_reveal(
            main, sentences,
            window=config.REVEAL_WINDOW_SECONDS,
            ceiling=_ceiling(duration_range),
        )
        reveal_applied = completed != main
        main = completed
        if ad_spans:
            cleared = ads.clamp_outside(main, ad_spans, min_seconds=2.0)
            if cleared is not None:
                main = cleared

    # The model's own setup choice wins; a local 8B model almost always says
    # the clip is self-contained, so fall back to detecting the dangling
    # reference ourselves.
    setup = _setup_span(lines, data, main[0])
    setup_source = "model" if setup else None
    if setup is None and config.STITCH_SETUP and words:
        setup = context.suggest_setup(
            words, main[0],
            lookback=config.SETUP_CONTEXT_SECONDS,
            max_seconds=config.SETUP_MAX_SECONDS,
        )
        setup_source = "auto" if setup else None

    spans = _compose_spans(main, setup, words, media_duration, duration_range,
                           ad_spans)
    # start/end describe the payoff, so dedupe and seeking stay meaningful even
    # when a setup span is prepended.
    start, end = spans[-1]
    sub = [l for l in lines
           if any(l["end"] > s and l["start"] < e for s, e in spans)]

    virality = _clamp(data.get("virality_score"), 0, 100, default=50)
    breakdown = {
        "hook_strength": _clamp(data.get("hook_strength"), 0, 10, default=5),
        "emotional_impact": _clamp(data.get("emotional_impact"), 0, 10, default=5),
        "standalone_clarity": _clamp(data.get("standalone_clarity"), 0, 10, default=5),
        "shareability": _clamp(data.get("shareability"), 0, 10, default=5),
    }

    return {
        **candidate,
        "start": round(start, 3),
        "end": round(end, 3),
        "spans": [{"start": s, "end": e} for s, e in spans],
        "stitched": len(spans) > 1,
        "setup_source": setup_source if len(spans) > 1 else None,
        "hook_anchored": hook_applied,
        "reveal_completed": reveal_applied,
        "hook_at": None if hook_line is None else round(float(hook_line["start"]), 3),
        "duration": round(timeline.total_duration(spans), 3),
        "text": " ".join(l["text"] for l in sub) or candidate["text"],
        "title": _clean_title(data.get("title")) or "Untitled clip",
        "hook": _as_text(data.get("hook")),
        "summary": _as_text(data.get("summary")),
        "reason": _as_text(data.get("reason")),
        "tags": _as_tags(data.get("tags")),
        "virality_score": virality,
        "breakdown": breakdown,
        # Retrieval agrees on *where* to look; the LLM judges *how good* it is.
        "final_score": round(0.75 * virality + 25.0 * candidate["retrieval_score"], 2),
        "scored_by": config.OLLAMA_MODEL,
    }


CRITIC_SYSTEM = """\
You are a ruthless short-form editor reviewing a cut someone else made. You \
answer only with a single JSON object."""

CRITIC_TEMPLATE = """\
Below are the numbered lines of a long-form transcript. The clip currently runs \
from line {start_line} to line {end_line}.

--- TRANSCRIPT ---
{lines}
--- END TRANSCRIPT ---

The clip's stated hook is: "{hook}"

Review the cut against three rules:

1. It must OPEN on the hook. Opening on a subscribe/like request, a price or \
spec recitation, or general admin talk is a failure.
2. It must CONTAIN THE PAYOFF. If the clip builds towards a reveal, it has to \
include the reveal, not stop on the promise of it.
3. It must contain no sponsor read or channel housekeeping.

If any rule fails, give the line numbers that would fix it. Keep the clip \
between {min_s:.0f} and {max_s:.0f} seconds. If the cut is already good, repeat \
its current line numbers.

Return exactly this JSON object:
{{
  "opens_on_hook": true or false,
  "contains_payoff": true or false,
  "has_housekeeping": true or false,
  "suggested_start_line": <integer line number>,
  "suggested_end_line": <integer line number>,
  "verdict": "one short sentence"
}}"""


def critique_clip(
    clip: dict[str, Any],
    lines: Sequence[dict[str, Any]],
) -> dict[str, Any] | None:
    """Ask the model to review a finished cut. Returns None if it cannot.

    Run over the handful of selected clips rather than every candidate, so the
    extra latency is a few calls instead of a dozen.
    """
    if not lines:
        return None

    start, end = float(clip["start"]), float(clip["end"])
    inside = [l for l in lines if l["end"] > start and l["start"] < end]
    if not inside:
        return None

    prompt = CRITIC_TEMPLATE.format(
        lines=_format_lines(lines, [l["n"] for l in inside]),
        start_line=inside[0]["n"],
        end_line=inside[-1]["n"],
        hook=_as_text(clip.get("hook"))[:200] or "(none given)",
        min_s=config.CLIP_MIN_SECONDS,
        max_s=config.CLIP_MAX_SECONDS,
    )

    from langchain_core.messages import HumanMessage, SystemMessage

    try:
        response = get_llm().invoke([
            SystemMessage(content=CRITIC_SYSTEM),
            HumanMessage(content=prompt),
        ])
    except Exception:      # noqa: BLE001 - the critic is optional by design
        return None

    content = response.content
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return _parse_json(str(content or ""))


# How much of the critic's complaint we can check ourselves. A hook sitting
# this close to the start is "opening on the hook" whatever the critic says.
HOOK_AT_START_SECONDS = 2.0
# Below this, an overlap with a sponsor read or subscribe ask is a rounding
# error from boundary snapping, not housekeeping the viewer would notice.
NOTICEABLE_FILLER_SECONDS = 1.0


def _clip_profile(
    spans: Sequence[tuple[float, float]],
    hook: dict[str, Any] | None,
    sentences: Sequence[dict[str, Any]],
    ad_spans: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """The few things about a cut we can judge without asking a model."""
    start, end = spans[0][0], spans[-1][1]
    inside = hook is not None and any(
        s <= float(hook["start"]) and float(hook["end"]) <= e + 0.5
        for s, e in spans
    )
    tail = next((s for s in reversed(sentences)
                 if float(s["end"]) <= end + 0.5), None)
    return {
        "hook_inside": inside,
        "hook_delay": (float(hook["start"]) - start) if inside else None,
        "ends_on_promise": bool(tail and narrative.promises_more(tail["text"])),
        "filler_seconds": sum(ads.overlap_seconds(s, (a["start"], a["end"]))
                              for s in spans for a in ad_spans),
        "duration": timeline.total_duration(list(spans)),
    }


def _supported_complaints(
    verdict: dict[str, Any],
    spans: Sequence[tuple[float, float]],
    hook: dict[str, Any] | None,
    sentences: Sequence[dict[str, Any]],
    ad_spans: Sequence[dict[str, Any]],
) -> set[str]:
    """Keep only the objections our own detectors do not contradict.

    A small local model will flag everything: one 17-minute job came back with
    the same three failures for all five finalists, "has_housekeeping" included
    for clips a minute clear of the nearest subscribe ask. Acting on that
    undoes the rules rather than correcting them.
    """
    profile = _clip_profile(spans, hook, sentences, ad_spans)
    complaints: set[str] = set()

    if not verdict["opens_on_hook"]:
        # Unlocatable hook = nothing to check it against, so the critic gets
        # the benefit of the doubt.
        delay = profile["hook_delay"]
        if delay is None or delay > HOOK_AT_START_SECONDS:
            complaints.add("opens_on_hook")

    if not verdict["contains_payoff"]:
        # We cannot tell a satisfying ending from an unsatisfying one, but we
        # can tell whether the clip stops on a promise or just short of a
        # reveal — which is what "missing payoff" means in practice.
        dangling = profile["ends_on_promise"] or (
            sentences and narrative.next_reveal(
                sentences, spans[-1][1], window=config.REVEAL_WINDOW_SECONDS,
            ) is not None
        )
        if dangling:
            complaints.add("contains_payoff")

    if verdict["has_housekeeping"]:
        if profile["filler_seconds"] >= NOTICEABLE_FILLER_SECONDS:
            complaints.add("has_housekeeping")

    return complaints


def _is_improvement(before: dict[str, Any], after: dict[str, Any]) -> bool:
    """Is the critic's recut better than the span the rules produced?"""
    if before["hook_inside"] and not after["hook_inside"]:
        return False
    if after["ends_on_promise"] and not before["ends_on_promise"]:
        return False
    if after["filler_seconds"] > before["filler_seconds"] + 0.5:
        return False

    delays = (before["hook_delay"], after["hook_delay"])
    if None not in delays and delays[1] > delays[0] + 1.0:
        return False       # pushed the hook further from the opening frame

    return (
        (after["hook_inside"] and not before["hook_inside"])
        or (None not in delays and delays[1] < delays[0] - 1.0)
        or (before["ends_on_promise"] and not after["ends_on_promise"])
        or after["filler_seconds"] < before["filler_seconds"] - 0.5
    )


def apply_critique(
    clip: dict[str, Any],
    critique: dict[str, Any] | None,
    lines: Sequence[dict[str, Any]],
    *,
    words: Sequence[dict[str, Any]] = (),
    media_duration: float | None = None,
    topic_boundaries: Sequence[float] = (),
    duration_range: tuple[float, float] | None = None,
    ad_spans: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    """Adopt the critic's bounds only if they measurably improve the clip.

    The critic is advisory, and on real footage it is unreliable in a specific
    way: `llama3.1:8b` returned the identical all-negative verdict for all five
    finalists of one job, including "has_housekeeping" for clips a minute away
    from the nearest subscribe ask. So a review is put through two filters
    before its bounds are considered at all:

    * **Corroboration** — every complaint it raises that we can measure
      ourselves is measured. A review whose every checkable claim is
      contradicted carries no information and is dropped.
    * **Improvement** — the proposal is profiled next to the span the rules
      produced (how far in the hook sits, whether it ends on a promise, how
      much filler it contains) and adopted only if it wins on something
      without losing on anything else.

    Legality alone is not enough: a three-line suggestion is grown to the
    minimum duration by `_compose_spans`, which turns a careless proposal into
    a technically valid clip built from bounds the critic never really chose.
    """
    if not critique or not lines:
        return clip

    verdict = {
        "opens_on_hook": bool(critique.get("opens_on_hook", True)),
        "contains_payoff": bool(critique.get("contains_payoff", True)),
        "has_housekeeping": bool(critique.get("has_housekeeping", False)),
        "verdict": _as_text(critique.get("verdict"))[:200],
        "applied": False,
    }
    current = [(float(s["start"]), float(s["end"]))
               for s in (clip.get("spans")
                         or [{"start": clip["start"], "end": clip["end"]}])]
    sentences = context.sentences(words) if words else []
    hook, _score = narrative.locate_hook(
        _as_text(clip.get("hook")), None, lines,
        threshold=config.HOOK_MATCH_THRESHOLD,
    )

    complaints = _supported_complaints(verdict, current, hook, sentences, ad_spans)
    verdict["upheld"] = sorted(complaints)
    if not complaints:
        # Either the critic is happy, or every objection it raised is one we
        # can measure and have measured as false.
        return {**clip, "critique": verdict}

    last = len(lines) - 1
    s_idx = _clamp_index(critique.get("suggested_start_line"), 0, last, default=0)
    e_idx = _clamp_index(critique.get("suggested_end_line"), 0, last, default=last)
    if e_idx < s_idx:
        s_idx, e_idx = e_idx, s_idx

    proposed = (float(lines[s_idx]["start"]), float(lines[e_idx]["end"]))
    # A window far under the minimum was not chosen as a clip; `_compose_spans`
    # would grow it to a legal length around bounds the critic never weighed.
    if proposed[1] - proposed[0] < config.CLIP_MIN_SECONDS * 0.5:
        return {**clip, "critique": verdict}

    spans = _compose_spans(proposed, None, words, media_duration,
                           duration_range, ad_spans)
    if not spans:
        return {**clip, "critique": verdict}

    total = timeline.total_duration(spans)
    if not (config.CLIP_MIN_SECONDS <= total <= config.CLIP_MAX_SECONDS):
        return {**clip, "critique": verdict}
    if ad_spans and ads.is_advert((spans[0][0], spans[-1][1]), ad_spans,
                                  threshold=config.AD_OVERLAP_THRESHOLD):
        return {**clip, "critique": verdict}

    # The recut has to earn its place. Observed failure: the critic reports
    # "does not contain the payoff", then proposes bounds that still stop on
    # the buildup and start 12s before the hook — legal, and worse than what
    # the rules produced.
    before = _clip_profile(current, hook, sentences, ad_spans)
    after = _clip_profile(spans, hook, sentences, ad_spans)
    if not _is_improvement(before, after):
        return {**clip, "critique": verdict}

    verdict["applied"] = True
    return {
        **clip,
        "start": round(spans[-1][0], 3),
        "end": round(spans[-1][1], 3),
        "spans": [{"start": s, "end": e} for s, e in spans],
        "stitched": len(spans) > 1,
        "duration": round(total, 3),
        "critique": verdict,
    }


def _invoke(prompt: str) -> str:
    from langchain_core.messages import HumanMessage, SystemMessage

    response = get_llm().invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])
    content = response.content
    if isinstance(content, list):  # some backends return content blocks
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content or "")


def _parse_json(raw: str) -> dict[str, Any] | None:
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None
    return None


#: How far outside the retrieved moment the main span may reach, in lines.
PAYOFF_SLACK = 2

#: A duration bucket may be exceeded by this fraction to reach a clean ending.
#: Ending mid-sentence to hit a length target exactly is the wrong trade.
BUCKET_TOLERANCE = 0.15


def _refine_bounds(
    lines: Sequence[dict[str, Any]],
    data: dict[str, Any],
    candidate: dict[str, Any],
    default_span: tuple[int, int] | None = None,
    duration_range: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """Apply the model's chosen line range, then force it back inside the limits.

    The choice is confined to the retrieved moment (plus a little slack). The
    prompt shows surrounding context so a setup span can be drawn from it, but
    an unconstrained main span lets the model wander off to whatever it finds
    most quotable in the window — which throws away retrieval's judgement and
    makes several candidates collapse onto the same passage.
    """
    last = len(lines) - 1
    # Falling back to the retrieved moment matters once `lines` carries extra
    # context: defaulting to 0..last would silently widen the clip to the whole
    # context window whenever the model omits the fields.
    fallback_start, fallback_end = default_span or (0, last)
    lo = max(0, min(fallback_start, fallback_end) - PAYOFF_SLACK)
    hi = min(last, max(fallback_start, fallback_end) + PAYOFF_SLACK)

    # A long clip has to be allowed to run forward past the retrieved window,
    # or it could never reach its requested length. Reaching *backward* stays
    # tightly limited, since that is what drifts off the retrieved moment.
    if duration_range:
        forward_limit = float(lines[lo]["start"]) + float(duration_range[1])
        while hi < last and float(lines[hi + 1]["end"]) <= forward_limit:
            hi += 1

    # These index into `lines`, so they must be ints: models occasionally emit
    # "start_line": 2.5, and a float subscript would raise TypeError.
    s_idx = _clamp_index(data.get("start_line"), lo, hi, default=fallback_start)
    e_idx = _clamp_index(data.get("end_line"), lo, hi, default=fallback_end)
    if e_idx < s_idx:
        s_idx, e_idx = e_idx, s_idx

    start = float(lines[s_idx]["start"])
    end = float(lines[e_idx]["end"])

    floor, ceiling = duration_range or (config.CLIP_MIN_SECONDS,
                                        config.CLIP_MAX_SECONDS)

    # Too short: extend forward line by line, then backward — but only inside
    # the payoff window, so padding never drifts into the context region.
    while end - start < floor and (e_idx < hi or s_idx > lo):
        if e_idx < hi:
            e_idx += 1
            end = float(lines[e_idx]["end"])
        elif s_idx > lo:
            s_idx -= 1
            start = float(lines[s_idx]["start"])

    # Too long: trim from the end, keeping the hook intact.
    while end - start > ceiling and e_idx > s_idx:
        e_idx -= 1
        end = float(lines[e_idx]["end"])

    if end <= start:
        start, end = float(candidate["start"]), float(candidate["end"])
    return start, end


def _finish_thought(
    span: tuple[float, float],
    topic_boundaries: Sequence[float],
    duration_range: tuple[float, float] | None,
    ad_spans: Sequence[dict[str, Any]] = (),
) -> tuple[float, float]:
    """Let a span run on to the end of the thought it started.

    A clean sentence ending can still stop mid-argument — "So there are three
    moves." is grammatical and leaves the viewer hanging.
    """
    if not config.TOPIC_COMPLETION or not topic_boundaries:
        return span
    finished = topics.complete_topic(
        span, topic_boundaries,
        max_extend=config.TOPIC_EXTEND_SECONDS,
        max_seconds=_ceiling(duration_range),
    )
    if ad_spans:
        cleared = ads.clamp_outside(finished, ad_spans, min_seconds=2.0)
        if cleared is not None:
            return cleared
    return finished


def _ceiling(duration_range: tuple[float, float] | None) -> float:
    """Longest a clip may run: its bucket plus a little slack for a clean end."""
    if not duration_range:
        return config.CLIP_MAX_SECONDS
    return min(duration_range[1] * (1.0 + BUCKET_TOLERANCE),
               config.CLIP_MAX_SECONDS)


def _setup_span(
    lines: Sequence[dict[str, Any]],
    data: dict[str, Any],
    main_start: float,
) -> tuple[float, float] | None:
    """The optional earlier passage the model asked to prepend, if it is sane.

    Rejected unless it sits wholly before the payoff: a "setup" that overlaps
    the main span would just replay the same audio twice.
    """
    if not config.STITCH_SETUP or not lines:
        return None
    raw_start, raw_end = data.get("setup_start_line"), data.get("setup_end_line")
    if raw_start is None or raw_end is None:
        return None

    last = len(lines) - 1
    s_idx = _clamp_index(raw_start, 0, last, default=0)
    e_idx = _clamp_index(raw_end, 0, last, default=0)
    if e_idx < s_idx:
        s_idx, e_idx = e_idx, s_idx

    start = float(lines[s_idx]["start"])
    end = float(lines[e_idx]["end"])
    if end > main_start:
        # Trim back to the payoff; if nothing useful is left, skip the setup.
        end = main_start
    if end - start < 1.0:
        return None

    # Keep the tail of the setup: the part nearest the payoff is what sets it up.
    if end - start > config.SETUP_MAX_SECONDS:
        start = end - config.SETUP_MAX_SECONDS
    return (start, end)


def _snap(span: tuple[float, float], words, media_duration, **limits) -> tuple[float, float]:
    """Move a span's edges onto natural pauses."""
    if not words:
        return span
    return boundaries.snap_span(
        span[0], span[1], words,
        max_extend=config.CLIP_SNAP_EXTEND,
        max_truncate=config.CLIP_SNAP_TRUNCATE,
        min_pause=config.CLIP_MIN_PAUSE,
        lead_in=config.CLIP_LEAD_IN_SECONDS,
        tail=config.CLIP_TAIL_SECONDS,
        media_duration=media_duration,
        **limits,
    )


def _compose_spans(
    main: tuple[float, float],
    setup: tuple[float, float] | None,
    words: Sequence[dict[str, Any]],
    media_duration: float | None,
    duration_range: tuple[float, float] | None = None,
    ad_spans: Sequence[dict[str, Any]] = (),
) -> list[tuple[float, float]]:
    """Snap the main span (and any setup), then fit them inside the length cap."""
    floor = (duration_range or (config.CLIP_MIN_SECONDS,))[0]
    ceiling = _ceiling(duration_range)
    spans = [_snap(main, words, media_duration,
                   min_seconds=floor, max_seconds=ceiling)]

    if setup:
        snapped = _snap(setup, words, media_duration,
                        max_seconds=config.SETUP_MAX_SECONDS)
        if snapped[1] > spans[0][0]:
            snapped = (snapped[0], spans[0][0])      # never overlap the payoff
        budget = ceiling - (spans[0][1] - spans[0][0])
        length = snapped[1] - snapped[0]
        if length > budget:
            # Keep the tail, nearest the payoff. `end - budget` is raw
            # arithmetic that lands wherever it lands, so move it onto a word
            # boundary — forwards only, since going back would blow the budget.
            trimmed = snapped[1] - budget
            if words:
                # Forward, not back: moving earlier would exceed the budget,
                # and the whole point is to avoid opening mid-word.
                trimmed = boundaries.align_forward(words, trimmed)
            snapped = (trimmed, snapped[1])
            length = snapped[1] - snapped[0]
        if length >= 1.0:
            spans.insert(0, snapped)

    # Cut out any sponsor read or subscribe ask that falls *inside* a span and
    # keep what is either side, so a clip is not thrown away — nor truncated to
    # its useless first few seconds — because of filler in the middle of it.
    if ad_spans:
        excised: list[tuple[float, float]] = []
        for span in spans:
            excised += ads.subtract(span, ad_spans, min_piece=3.0)
        if excised:
            # Re-snap: subtraction cuts at the filler's edge, which is not
            # necessarily a word or sentence boundary.
            spans = [
                _snap(s, words, media_duration, max_seconds=ceiling)
                for s in excised
            ]
            # Excision can leave a fragment: a clip anchored back into a
            # sponsor read once came out as a 4.5s scrap. Keep the largest
            # piece and let the minimum-length rule grow it instead.
            if timeline.total_duration(spans) < floor:
                longest = max(spans, key=lambda s: s[1] - s[0])
                spans = [_snap(longest, words, media_duration,
                               min_seconds=floor, max_seconds=ceiling)]
                # Growing it may have reached back into the excluded region.
                regrown = ads.subtract(spans[0], ad_spans, min_piece=3.0)
                if regrown:
                    spans = [max(regrown, key=lambda s: s[1] - s[0])]

    return timeline.merge_touching(spans)


def _fallback(
    candidate: dict[str, Any],
    why: str,
    *,
    words: Sequence[dict[str, Any]] = (),
    media_duration: float | None = None,
) -> dict[str, Any]:
    """Keep the pipeline moving when the LLM is unavailable or misbehaving."""
    span = _snap(
        (float(candidate["start"]), float(candidate["end"])), words, media_duration,
        min_seconds=config.CLIP_MIN_SECONDS, max_seconds=config.CLIP_MAX_SECONDS,
    )
    return {
        **candidate,
        "start": round(span[0], 3),
        "end": round(span[1], 3),
        "spans": [{"start": span[0], "end": span[1]}],
        "stitched": False,
        "setup_source": None,
        "duration": round(span[1] - span[0], 3),
        "title": _clean_title(candidate["text"][:60]) or "Untitled clip",
        "hook": candidate["text"][:140],
        "summary": "",
        "reason": f"Retrieval-only score ({why}).",
        "tags": [],
        "virality_score": round(candidate["retrieval_score"] * 100, 1),
        "breakdown": {},
        "final_score": round(candidate["retrieval_score"] * 100, 2),
        "scored_by": "retrieval-fallback",
    }


def _clamp_index(value: Any, lo: int, hi: int, default: int) -> int:
    """Clamp to an integer list index, tolerating floats and numeric strings."""
    try:
        num = int(round(float(value)))
    except (TypeError, ValueError):
        num = int(default)
    return max(lo, min(hi, num))


def _clamp(value: Any, lo: float, hi: float, default: float) -> float | int:
    try:
        num = float(value)
    except (TypeError, ValueError):
        num = float(default)
    num = max(lo, min(hi, num))
    return int(num) if float(num).is_integer() else round(num, 2)


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return "" if value is None else str(value).strip()


def _as_tags(value: Any) -> list[str]:
    if isinstance(value, str):
        value = re.split(r"[,;]", value)
    if not isinstance(value, list):
        return []
    tags = [str(t).strip().lstrip("#").lower() for t in value]
    return [t for t in tags if t][:5]


def _clean_title(value: Any) -> str:
    title = _as_text(value).strip("\"' ")
    title = re.sub(r"\s+", " ", title)
    return title[:80]
