"""End-to-end orchestration: ingest → transcribe → index → retrieve → score → render.

Transcription and indexing are deliberately overlapped. Whisper hands each
finished segment to a window builder on the decode thread; completed windows go
onto a queue that a background indexer drains into ChromaDB. Embedding latency
therefore never blocks the decoder, and the vector index is essentially ready
the moment transcription ends.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path
from typing import Any

import config
from app.core import (
    boundaries, ingest, render, scoring, timeline, topics, transcribe,
)
from app.core.chunker import WindowBuilder
from app.core.vectorstore import StreamingIndex
from app.jobs import Job, JobCancelled, StageReporter, store

_SENTINEL = object()


class BackgroundIndexer:
    """Drains completed windows into Chroma on its own thread."""

    def __init__(self, index: StreamingIndex) -> None:
        self.index = index
        self.queue: queue.Queue = queue.Queue()
        self.indexed = 0
        self.error: BaseException | None = None
        self._thread = threading.Thread(target=self._loop, name="indexer", daemon=True)

    def start(self) -> "BackgroundIndexer":
        self._thread.start()
        return self

    def submit(self, chunk: dict[str, Any]) -> None:
        self.queue.put(chunk)

    def close(self) -> int:
        """Signal end-of-stream, wait for the drain, return chunks indexed."""
        self.queue.put(_SENTINEL)
        self._thread.join()
        if self.error:
            raise self.error
        return self.indexed

    def _loop(self) -> None:
        try:
            while True:
                item = self.queue.get()
                if item is _SENTINEL:
                    self.indexed += self.index.flush()
                    return
                self.indexed += self.index.add(item)
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller's thread
            self.error = exc


def process(job: Job) -> None:
    """Run the whole pipeline for `job`, mutating it as each stage completes."""
    t0 = time.time()

    # --- 1. ingest ---
    media_info = ingest.run(job)
    store.update(job.id, media={**job.media, **media_info},
                 message=f"Ingested {media_info['duration'] / 60:.1f} min of media")
    _check(job)

    # --- 2 & 3. transcribe, streaming windows into the vector index ---
    index = StreamingIndex(job.id)
    builder = WindowBuilder(max_seconds=config.WINDOW_MAX_SECONDS)
    indexer = BackgroundIndexer(index).start()
    windows: list[dict[str, Any]] = []

    try:
        with StageReporter(job, "transcribe") as rep:
            rep.log(f"Loading Whisper `{config.WHISPER_MODEL}` ({config.WHISPER_COMPUTE_TYPE})…")

            def on_segment(segment: dict[str, Any]) -> None:
                for window in builder.push(segment):
                    windows.append(window)
                    indexer.submit(window)

            segments, meta = transcribe.transcribe(
                media_info["audio_path"],
                media_info["duration"],
                on_segment=on_segment,
                on_progress=lambda frac, msg: rep.progress(
                    frac, f"{msg} · {len(windows)} windows queued"
                ),
                should_cancel=lambda: job.cancelled,
            )
            _check(job)
            if not segments:
                raise ValueError("Whisper produced no speech — is there audible dialogue?")
            rep.log(
                f"Transcribed {meta['segment_count']} segments / {meta['word_count']} words "
                f"({meta['language']}, p={meta['language_probability']})"
            )
            store.update(job.id, transcript=segments)

        with StageReporter(job, "index") as rep:
            rep.progress(0.3, "Flushing final windows…")
            for window in builder.finish():
                windows.append(window)
                indexer.submit(window)
            indexed = indexer.close()
            rep.progress(1.0, f"Indexed {indexed} windows in ChromaDB")
            rep.log(f"ChromaDB collection `{index.collection_name}` holds {indexed} vectors")
    finally:
        # Never leave the indexer thread hanging if a stage above blew up.
        if indexer._thread.is_alive():
            try:
                indexer.close()
            except Exception:  # noqa: BLE001
                pass

    if index.count == 0:
        raise ValueError("Nothing was indexed — transcript may be empty.")
    _check(job)

    store.update(job.id, stats={
        **job.stats,
        **meta,
        "windows": len(windows),
        "vectors": index.count,
        "duration": media_info["duration"],
    })

    # --- 4. retrieve ---
    with StageReporter(job, "retrieve") as rep:
        rep.progress(0.2, "Probing the index for high-signal moments…")
        candidates = scoring.retrieve_candidates(index)
        if not candidates:
            raise ValueError("Retrieval returned no candidates.")
        rep.progress(1.0, f"{len(candidates)} candidate moments retrieved")
        rep.log(
            f"Top candidate: {candidates[0]['start']:.0f}s "
            f"(relevance {candidates[0]['relevance']:.2f}, "
            f"{candidates[0]['match_count']} probe hits)"
        )
    _check(job)

    # --- 5. score ---
    # Word timings let the scorer move cut points onto real pauses, and are
    # what keeps clips from ending on a chopped-off word.
    words = boundaries.flatten_words(segments)

    # Where each thought finishes, so a clip can end on a completed point
    # rather than merely a completed sentence. The embedding pass reuses the
    # index's embedder; markers alone still work if it is unavailable.
    topic_boundaries: list[float] = []
    if config.TOPIC_COMPLETION:
        topic_boundaries = topics.find_boundaries(
            words,
            embed=getattr(index, "embed_texts", None),
            threshold=config.TOPIC_SHIFT_THRESHOLD,
        )

    # Each slot aims at its own slice of the allowed length range, otherwise
    # every clip comes back about the same length.
    buckets = (
        scoring.duration_buckets(config.MAX_CLIPS)
        if config.DURATION_SPREAD else []
    )

    scored: list[dict[str, Any]] = []
    with StageReporter(job, "score") as rep:
        if buckets:
            rep.log("Length targets: " + ", ".join(
                f"{lo:.0f}-{hi:.0f}s" for lo, hi in buckets))
        if topic_boundaries:
            rep.log(f"{len(topic_boundaries)} topic boundaries detected")
        for i, candidate in enumerate(candidates):
            _check(job)
            rep.progress(
                i / len(candidates),
                f"Scoring candidate {i + 1}/{len(candidates)} with {config.OLLAMA_MODEL}…",
            )
            try:
                bucket = buckets[i % len(buckets)] if buckets else None
                result = scoring.score_candidate(
                    candidate,
                    # Padded behind so the model can pull a setup passage, and
                    # ahead by the bucket's ceiling so a long clip has room.
                    scoring.build_lines(
                        segments,
                        candidate["start"] - config.SETUP_CONTEXT_SECONDS,
                        candidate["end"] + max(10.0, bucket[1] if bucket else 10.0),
                    ),
                    words=words,
                    media_duration=media_info["duration"],
                    topic_boundaries=topic_boundaries,
                    duration_range=bucket,
                )
            except Exception as exc:  # noqa: BLE001 - one bad candidate must not kill the run
                rep.log(f"Candidate {i + 1} failed to score: {exc}")
                continue
            scored.append(result)
            rep.log(f"[{result['virality_score']:>3}] {result['title']}")

        if not scored:
            raise ValueError("No candidate could be scored — is Ollama running?")

        scored.sort(key=lambda c: c["final_score"], reverse=True)
        deduped = scoring.dedupe_spans(scored, max_overlap=0.35)
        selected = (
            scoring.spread_by_duration(deduped, config.MAX_CLIPS)
            if config.DURATION_SPREAD else deduped[: config.MAX_CLIPS]
        )
        rep.progress(1.0, f"Selected {len(selected)} clips — " + ", ".join(
            f"{c['duration']:.0f}s" for c in selected))
    _check(job)

    # --- 6. render ---
    clip_dir = config.CLIP_DIR / job.id
    clip_dir.mkdir(parents=True, exist_ok=True)
    clips: list[dict[str, Any]] = []

    with StageReporter(job, "render") as rep:
        total = len(selected)
        for i, pick in enumerate(selected):
            _check(job)
            clip_id = f"clip-{i + 1:02d}"
            rep.progress(i / total, f"Rendering {clip_id} of {total} — “{pick['title']}”")

            spans = pick.get("spans") or [(pick["start"], pick["end"])]
            info = render.render_clip(
                media_info["video_path"],
                clip_dir / f"{clip_id}.mp4",
                spans,
                segments,
                has_video=media_info["has_video"],
                on_progress=lambda frac, i=i: rep.progress((i + frac) / total),
            )

            clip = {
                "id": clip_id,
                "rank": i + 1,
                "job_id": job.id,
                "start": pick["start"],
                "end": pick["end"],
                "duration": pick["duration"],
                "title": pick["title"],
                "hook": pick["hook"],
                "summary": pick["summary"],
                "reason": pick["reason"],
                "tags": pick["tags"],
                "text": pick["text"],
                "virality_score": pick["virality_score"],
                "final_score": pick["final_score"],
                "breakdown": pick["breakdown"],
                "relevance": pick["relevance"],
                "matched_queries": pick["matched_queries"],
                "scored_by": pick["scored_by"],
                "video_url": f"/api/jobs/{job.id}/clips/{clip_id}/file",
                "thumbnail_url": f"/api/jobs/{job.id}/clips/{clip_id}/thumbnail",
                "srt_url": f"/api/jobs/{job.id}/clips/{clip_id}/srt",
                **{k: info[k] for k in ("filename", "size_bytes", "width", "height",
                                        "rendered_duration", "path", "thumbnail",
                                        "subtitles_srt", "captions_burned",
                                        "caption_backend", "spans", "stitched",
                                        "fill", "crop", "subject_x")},
            }
            if info.get("captions_skipped_reason") and i == 0:
                rep.log(f"Captions not burned in — {info['captions_skipped_reason']}. "
                        f"Sidecar .srt is still written next to each clip.")

            clips.append(clip)
            # Publish incrementally so the UI fills in as each clip lands.
            store.update(job.id, clips=list(clips))
            stitch_note = ""
            if info["stitched"]:
                pieces = " + ".join(f"{s['start']:.0f}-{s['end']:.0f}s"
                                    for s in info["spans"])
                stitch_note = f", stitched from {pieces}"
            fill_note = info["fill"]
            if info["fill"] == "crop" and info["subject_x"] is not None:
                fill_note += f" @ {info['subject_x'] * 100:.0f}% width"
            rep.log(f"Rendered {clip_id} — {info['width']}x{info['height']}, "
                    f"{info['size_bytes'] / 1e6:.1f} MB, "
                    f"captions via {info['caption_backend']}, "
                    f"{fill_note}{stitch_note}")

        rep.progress(1.0, f"{len(clips)} clips rendered")

    store.update(job.id, stats={
        **job.stats,
        "candidates_retrieved": len(candidates),
        "candidates_scored": len(scored),
        "clips": len(clips),
        "elapsed_seconds": round(time.time() - t0, 1),
    })
    persist(job)


def _check(job: Job) -> None:
    if job.cancelled:
        raise JobCancelled(job.id)


# --- persistence so finished jobs survive a server restart ---

def result_path(job_id: str) -> Path:
    return config.JOB_DIR / f"{job_id}.json"


def persist(job: Job) -> None:
    payload = job.to_dict(include_transcript=True)
    payload["status"] = "done"
    result_path(job.id).write_text(json.dumps(payload, default=str), encoding="utf-8")


def restore_all() -> int:
    """Reload previously completed jobs into the in-memory store at startup."""
    restored = 0
    for path in sorted(config.JOB_DIR.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        job = Job(
            id=payload["id"],
            source=payload.get("source", "upload"),
            source_label=payload.get("source_label", ""),
            stage_progress=1.0,
        )
        for field in ("status", "stage", "message", "created_at", "started_at",
                      "finished_at", "media", "stats", "clips", "transcript"):
            if payload.get(field) is not None:
                setattr(job, field, payload[field])
        store.adopt(job)
        restored += 1
    return restored
