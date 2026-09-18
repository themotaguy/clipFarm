#!/usr/bin/env python
"""Score clip quality across every saved job, so "better" is measurable.

Clip selection is a pile of heuristics stacked on a small local model, and the
only way to tell whether a change helped is to measure it on real transcripts
rather than eyeball one clip. This replays the *current* rules over each saved
job's transcript and reports how the clips come out.

It deliberately re-derives spans instead of grading the spans stored in the
job file: those were produced by whatever the code looked like at the time, so
grading them would measure history, not the code in front of you.

    python scripts/quality_report.py              # every saved job
    python scripts/quality_report.py --job <id>   # just one
    python scripts/quality_report.py --json       # machine readable
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from app.core import ads, boundaries, context, narrative, scoring, timeline, topics  # noqa: E402


def _replay(job: dict[str, Any]) -> list[dict[str, Any]]:
    """Re-derive each stored clip's span using the rules as they stand now."""
    transcript = job.get("transcript") or []
    words = boundaries.flatten_words(transcript)
    if not words or not job.get("clips"):
        return []

    sentences = context.sentences(words)
    topic_boundaries = topics.find_boundaries(words)
    excluded = ads.find_excluded_spans(
        words,
        gap_seconds=config.AD_GAP_SECONDS,
        min_seconds=config.AD_MIN_SECONDS,
        filler=config.FILLER_DETECTION,
    )
    duration = float(job.get("stats", {}).get("duration") or 0.0)

    rows: list[dict[str, Any]] = []
    for clip in job["clips"]:
        stored = clip.get("spans") or [{"start": clip["start"], "end": clip["end"]}]
        span = (float(stored[0]["start"]), float(stored[-1]["end"]))

        # Mirror the pipeline, which discards candidates inside excluded
        # content before scoring them. Older saved jobs predate ad detection
        # and contain clips cut straight from a sponsor read; grading those
        # would measure the old code, not this one.
        if excluded and ads.is_advert(span, excluded,
                                      threshold=config.AD_OVERLAP_THRESHOLD):
            continue
        lines = scoring.build_lines(
            transcript, span[0] - config.SETUP_CONTEXT_SECONDS, span[1] + 90.0
        )

        hook, score = narrative.locate_hook(
            clip.get("hook") or "", None, lines,
            threshold=config.HOOK_MATCH_THRESHOLD,
        )
        if config.HOOK_ANCHOR and hook is not None:
            max_trim = (config.HOOK_TRIM_STRONG_SECONDS
                        if score >= config.HOOK_STRONG_MATCH
                        else config.HOOK_TRIM_MAX_SECONDS)
            span = narrative.anchor_to_hook(
                span, hook, run_up=config.HOOK_RUN_UP_SECONDS,
                max_trim=max_trim, ceiling=config.CLIP_MAX_SECONDS,
            )
        if config.TOPIC_COMPLETION and topic_boundaries:
            span = topics.complete_topic(
                span, topic_boundaries,
                max_extend=config.TOPIC_EXTEND_SECONDS,
                max_seconds=config.CLIP_MAX_SECONDS,
            )
        if config.REVEAL_COMPLETION:
            span = narrative.complete_reveal(
                span, sentences, window=config.REVEAL_WINDOW_SECONDS,
                ceiling=config.CLIP_MAX_SECONDS,
            )
        spans = scoring._compose_spans(span, None, words, duration, None, excluded)
        if not spans:
            continue

        rows.append(_grade(spans, words, sentences, excluded, hook))
    return rows


def _grade(
    spans: list[tuple[float, float]],
    words: list[dict[str, Any]],
    sentences: list[dict[str, Any]],
    excluded: list[dict[str, Any]],
    hook: dict[str, Any] | None,
) -> dict[str, Any]:
    start, end = spans[0][0], spans[-1][1]
    total = timeline.total_duration(spans)

    last = max((w for w in words if float(w["end"]) <= end + 1e-6),
               key=lambda w: float(w["end"]), default=None)
    tail = next((s for s in reversed(sentences)
                 if float(s["end"]) <= end + 0.5), None)

    return {
        "duration": round(total, 1),
        "under_minimum": total < config.CLIP_MIN_SECONDS - 0.5,
        "ends_on_sentence": bool(last and boundaries.ends_sentence(last)),
        "ends_on_promise": bool(tail and narrative.promises_more(tail["text"])),
        "mid_word_cut": any(
            boundaries._straddling_word(words, edge)
            for span in spans for edge in span
        ),
        # Measured per span, not across the envelope: a stitched clip that
        # correctly excised its filler has a gap where the filler used to be,
        # and grading the envelope would count the very thing we removed.
        "excluded_overlap": round(
            (sum(ads.overlap_fraction(s, excluded) * (s[1] - s[0]) for s in spans)
             / total * 100) if total > 0 else 0.0,
            1,
        ),
        "hook_found": hook is not None,
        "seconds_to_hook": (
            None if hook is None else round(float(hook["start"]) - start, 1)
        ),
    }


def report(paths: list[str], as_json: bool) -> int:
    jobs: list[dict[str, Any]] = []
    for path in paths:
        try:
            job = json.loads(Path(path).read_text())
        except (OSError, json.JSONDecodeError):
            continue
        rows = _replay(job)
        if rows:
            jobs.append({"job": Path(path).stem,
                         "minutes": round(job.get("stats", {}).get("duration", 0) / 60, 1),
                         "clips": rows})

    if as_json:
        print(json.dumps(jobs, indent=2))
        return 0
    if not jobs:
        print("No saved jobs with transcripts found under data/jobs/.")
        return 1

    header = (f"{'job':14}{'min':>6}{'clips':>7}{'sentence':>10}{'no promise':>12}"
              f"{'clean cut':>11}{'no filler':>11}{'>=min':>8}{'hook@':>8}"
              f"{'lengths':>22}")
    print(header)
    print("-" * len(header))

    totals = {k: 0 for k in ("n", "sentence", "no_promise", "clean",
                             "no_filler", "long_enough")}
    hook_delays: list[float] = []
    for entry in jobs:
        rows = entry["clips"]
        n = len(rows)
        sentence = sum(r["ends_on_sentence"] for r in rows)
        no_promise = sum(not r["ends_on_promise"] for r in rows)
        clean = sum(not r["mid_word_cut"] for r in rows)
        no_filler = sum(r["excluded_overlap"] < 25.0 for r in rows)
        long_enough = sum(not r["under_minimum"] for r in rows)
        delays = [r["seconds_to_hook"] for r in rows if r["seconds_to_hook"] is not None]
        hook_delays += delays
        hook_txt = f"{sum(delays) / len(delays):.1f}s" if delays else "-"
        lengths = ",".join(f"{r['duration']:.0f}" for r in sorted(
            rows, key=lambda r: r["duration"]))

        totals["n"] += n
        totals["sentence"] += sentence
        totals["no_promise"] += no_promise
        totals["clean"] += clean
        totals["no_filler"] += no_filler
        totals["long_enough"] += long_enough
        print(f"{entry['job'][:13]:14}{entry['minutes']:>6}{n:>7}"
              f"{f'{sentence}/{n}':>10}{f'{no_promise}/{n}':>12}"
              f"{f'{clean}/{n}':>11}{f'{no_filler}/{n}':>11}"
              f"{f'{long_enough}/{n}':>8}{hook_txt:>8}{lengths:>22}")

    count = totals["n"]
    sentence_txt = f"{totals['sentence']}/{count}"
    promise_txt = f"{totals['no_promise']}/{count}"
    clean_txt = f"{totals['clean']}/{count}"
    filler_txt = f"{totals['no_filler']}/{count}"
    hook_txt = (f"{sum(hook_delays) / len(hook_delays):.1f}s"
                if hook_delays else "-")

    print("-" * len(header))
    min_txt = f"{totals['long_enough']}/{count}"
    print(f"{'TOTAL':14}{'':>6}{count:>7}{sentence_txt:>10}{promise_txt:>12}"
          f"{clean_txt:>11}{filler_txt:>11}{min_txt:>8}{hook_txt:>8}")
    print()
    print("sentence   = ends on a full stop        no promise = does not end on a buildup")
    print("clean cut  = no boundary inside a word  no filler  = <25% sponsor/housekeeping")
    print(">=min      = at least CLIP_MIN_SECONDS long")
    print("hook@      = mean seconds from clip start to its hook (lower is better)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", help="grade a single job id")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    args = parser.parse_args()

    if args.job:
        paths = [str(config.JOB_DIR / f"{args.job}.json")]
    else:
        paths = sorted(glob.glob(str(config.JOB_DIR / "*.json")))
    return report(paths, args.json)


if __name__ == "__main__":
    raise SystemExit(main())
