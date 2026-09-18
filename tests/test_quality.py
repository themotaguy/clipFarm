"""Quality thresholds over the real saved transcripts.

These are the guard rails for clip selection. The individual rules have unit
tests; this measures what they add up to on genuine footage, so a change that
looks harmless in isolation but degrades real clips fails here.

Skipped when there are no saved jobs — a fresh clone has none.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import pytest

import config

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

quality_report = pytest.importorskip("quality_report")


def _jobs() -> list[dict]:
    out = []
    for path in sorted(glob.glob(str(config.JOB_DIR / "*.json"))):
        try:
            job = json.loads(Path(path).read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if job.get("transcript") and job.get("clips"):
            out.append(job)
    return out


@pytest.fixture(scope="module")
def graded() -> list[dict]:
    jobs = _jobs()
    if not jobs:
        pytest.skip("no saved jobs to grade")
    rows: list[dict] = []
    for job in jobs:
        rows += quality_report._replay(job)
    if not rows:
        pytest.skip("saved jobs produced no gradeable clips")
    return rows


def _share(rows: list[dict], key: str, want: bool = True) -> float:
    return sum(1 for r in rows if bool(r[key]) is want) / len(rows)


def test_almost_every_clip_ends_on_a_sentence(graded):
    share = _share(graded, "ends_on_sentence")
    assert share >= 0.90, f"only {share:.0%} of clips end on a full stop"


def test_no_clip_ends_on_an_unresolved_buildup(graded):
    """The reported failure: ending on "the one more thing we all saw coming"."""
    offenders = [r for r in graded if r["ends_on_promise"]]
    assert not offenders, f"{len(offenders)} clip(s) end on a promise"


def test_no_clip_is_cut_inside_a_word(graded):
    offenders = [r for r in graded if r["mid_word_cut"]]
    assert not offenders, f"{len(offenders)} clip(s) have a boundary inside a word"


def test_clips_are_not_mostly_sponsor_or_housekeeping(graded):
    offenders = [r for r in graded if r["excluded_overlap"] >= 25.0]
    assert not offenders, f"{len(offenders)} clip(s) are mostly excluded content"


def test_hooks_are_found_and_near_the_start(graded):
    located = [r for r in graded if r["seconds_to_hook"] is not None]
    assert len(located) / len(graded) >= 0.75, "hook rarely locatable"
    # The complaint was clips opening 14s+ of filler before the hook.
    mean_delay = sum(r["seconds_to_hook"] for r in located) / len(located)
    assert mean_delay <= 8.0, f"hook lands {mean_delay:.1f}s into the clip on average"


def test_clip_lengths_stay_inside_the_configured_range(graded):
    for row in graded:
        assert row["duration"] <= config.CLIP_MAX_SECONDS + 1.0, row


def test_most_clips_meet_the_minimum_length(graded):
    share = _share(graded, "under_minimum", want=False)
    assert share >= 0.90, f"{1 - share:.0%} of clips are under the minimum length"
