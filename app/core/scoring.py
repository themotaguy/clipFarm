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

Also choose the tightest in/out points: pick the line where the clip should START \
(ideally the strongest hook, cutting throat-clearing preamble) and the line where \
it should END (a clean resolution, not mid-thought). The resulting clip must be \
between {min_s:.0f} and {max_s:.0f} seconds long.

Return exactly this JSON shape:
{{
  "title": "punchy clip title, max 60 characters, no quotes around it",
  "hook": "the single most scroll-stopping sentence from the excerpt, verbatim",
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


def _format_lines(lines: Sequence[dict[str, Any]]) -> str:
    return "\n".join(
        f"[{l['n']}] ({l['start']:.1f}s–{l['end']:.1f}s) {l['text']}" for l in lines
    )


def score_candidate(
    candidate: dict[str, Any],
    lines: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Grade one candidate with Ollama and refine its in/out points."""
    if not lines:
        return _fallback(candidate, "no transcript lines in range")

    prompt = USER_TEMPLATE.format(
        lines=_format_lines(lines),
        min_s=config.CLIP_MIN_SECONDS,
        max_s=config.CLIP_MAX_SECONDS,
    )

    raw = _invoke(prompt)
    data = _parse_json(raw)
    if data is None:
        # One retry with a blunter instruction before giving up on the LLM.
        raw = _invoke(prompt + "\n\nOutput ONLY the JSON object. No prose, no markdown.")
        data = _parse_json(raw)
    if data is None:
        return _fallback(candidate, "model did not return parseable JSON")

    start, end = _refine_bounds(lines, data, candidate)
    sub = [l for l in lines if l["end"] > start and l["start"] < end]

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
        "duration": round(end - start, 3),
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


def _refine_bounds(
    lines: Sequence[dict[str, Any]],
    data: dict[str, Any],
    candidate: dict[str, Any],
) -> tuple[float, float]:
    """Apply the model's chosen line range, then force it back inside the duration limits."""
    last = len(lines) - 1
    s_idx = _clamp(data.get("start_line"), 0, last, default=0)
    e_idx = _clamp(data.get("end_line"), 0, last, default=last)
    if e_idx < s_idx:
        s_idx, e_idx = e_idx, s_idx

    start = float(lines[s_idx]["start"])
    end = float(lines[e_idx]["end"])

    # Too short: extend forward line by line, then backward.
    while end - start < config.CLIP_MIN_SECONDS and (e_idx < last or s_idx > 0):
        if e_idx < last:
            e_idx += 1
            end = float(lines[e_idx]["end"])
        elif s_idx > 0:
            s_idx -= 1
            start = float(lines[s_idx]["start"])

    # Too long: trim from the end, keeping the hook intact.
    while end - start > config.CLIP_MAX_SECONDS and e_idx > s_idx:
        e_idx -= 1
        end = float(lines[e_idx]["end"])

    if end <= start:
        start, end = float(candidate["start"]), float(candidate["end"])
    return start, end


def _fallback(candidate: dict[str, Any], why: str) -> dict[str, Any]:
    """Keep the pipeline moving when the LLM is unavailable or misbehaving."""
    return {
        **candidate,
        "duration": round(candidate["end"] - candidate["start"], 3),
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
