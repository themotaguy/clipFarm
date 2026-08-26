"""In-process job registry with a per-job pub/sub bus that backs the SSE stream.

Jobs run on daemon worker threads. Every state mutation is published to any
listener subscribed to that job, so the browser sees stage transitions and
progress the moment they happen instead of polling.
"""

from __future__ import annotations

import json
import queue
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

import config

# Ordered pipeline stages. The UI renders these as a progress rail.
STAGES = [
    ("ingest", "Fetching & normalizing media"),
    ("transcribe", "Transcribing with Whisper"),
    ("index", "Streaming chunks into ChromaDB"),
    ("retrieve", "Semantic retrieval of candidates"),
    ("score", "Scoring virality with Ollama"),
    ("render", "Rendering vertical clips"),
]
STAGE_KEYS = [k for k, _ in STAGES]


@dataclass
class Job:
    id: str
    source: str                      # "upload" | "url"
    source_label: str                # filename or URL
    status: str = "queued"           # queued | running | done | error | cancelled
    stage: str | None = None
    stage_progress: float = 0.0      # 0..1 within the current stage
    message: str = "Queued"
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    # Filled in as the pipeline advances.
    media: dict[str, Any] = field(default_factory=dict)
    transcript: list[dict[str, Any]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    clips: list[dict[str, Any]] = field(default_factory=list)

    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def overall_progress(self) -> float:
        """Fraction of the whole pipeline complete, weighted evenly by stage."""
        if self.status == "done":
            return 1.0
        if self.stage is None:
            return 0.0
        try:
            idx = STAGE_KEYS.index(self.stage)
        except ValueError:
            return 0.0
        return (idx + min(max(self.stage_progress, 0.0), 1.0)) / len(STAGE_KEYS)

    def to_dict(self, *, include_transcript: bool = False) -> dict[str, Any]:
        payload = {
            "id": self.id,
            "source": self.source,
            "source_label": self.source_label,
            "status": self.status,
            "stage": self.stage,
            "stage_progress": round(self.stage_progress, 4),
            "progress": round(self.overall_progress(), 4),
            "message": self.message,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "media": self.media,
            "stats": self.stats,
            "clips": self.clips,
            "stages": [{"key": k, "label": lbl} for k, lbl in STAGES],
        }
        if include_transcript:
            payload["transcript"] = self.transcript
        return payload


class JobStore:
    """Thread-safe job registry + fan-out event bus."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._subscribers: dict[str, list[queue.Queue]] = {}
        self._lock = threading.RLock()

    # --- registry ---

    def create(self, source: str, source_label: str) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], source=source, source_label=source_label)
        with self._lock:
            self._jobs[job.id] = job
            self._subscribers[job.id] = []
        return job

    def adopt(self, job: Job) -> Job:
        """Insert an already-constructed job (used when rehydrating from disk)."""
        with self._lock:
            self._jobs[job.id] = job
            self._subscribers.setdefault(job.id, [])
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def delete(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.pop(job_id, None)
            if job is None:
                return False
            job._cancel.set()
            for q in self._subscribers.pop(job_id, []):
                q.put(None)
        return True

    def cancel(self, job_id: str) -> bool:
        job = self.get(job_id)
        if job is None or job.status in {"done", "error", "cancelled"}:
            return False
        job._cancel.set()
        self.update(job_id, message="Cancelling…")
        return True

    # --- events ---

    def subscribe(self, job_id: str) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=512)
        with self._lock:
            self._subscribers.setdefault(job_id, []).append(q)
        return q

    def unsubscribe(self, job_id: str, q: queue.Queue) -> None:
        with self._lock:
            subs = self._subscribers.get(job_id)
            if subs and q in subs:
                subs.remove(q)

    def _publish(self, job_id: str, event: dict[str, Any]) -> None:
        with self._lock:
            subs = list(self._subscribers.get(job_id, []))
        for q in subs:
            try:
                q.put_nowait(event)
            except queue.Full:
                # A wedged listener must never stall the pipeline thread.
                pass

    def update(self, job_id: str, **fields: Any) -> Job | None:
        """Patch job fields and broadcast the new state."""
        job = self.get(job_id)
        if job is None:
            return None
        with self._lock:
            for key, value in fields.items():
                if hasattr(job, key):
                    setattr(job, key, value)
        self._publish(job_id, {"type": "state", "job": job.to_dict()})
        return job

    def log(self, job_id: str, message: str) -> None:
        """Emit a human-readable line without changing job state."""
        self._publish(job_id, {"type": "log", "message": message, "ts": time.time()})

    # --- execution ---

    def run(self, job: Job, target: Callable[[Job], None]) -> None:
        def _wrapper() -> None:
            self.update(job.id, status="running", started_at=time.time(),
                        stage=STAGE_KEYS[0], message="Starting…")
            try:
                target(job)
                if job.cancelled:
                    self.update(job.id, status="cancelled", finished_at=time.time(),
                                message="Cancelled")
                else:
                    self.update(job.id, status="done", finished_at=time.time(),
                                stage_progress=1.0,
                                message=f"Done — {len(job.clips)} clip(s) ready")
            except JobCancelled:
                self.update(job.id, status="cancelled", finished_at=time.time(),
                            message="Cancelled")
            except Exception as exc:  # noqa: BLE001 - surface any failure to the client
                detail = traceback.format_exc(limit=6)
                self.log(job.id, detail)
                self.update(job.id, status="error", finished_at=time.time(),
                            error=f"{type(exc).__name__}: {exc}", message="Failed")
            finally:
                self._publish(job.id, {"type": "end"})

        threading.Thread(target=_wrapper, name=f"job-{job.id}", daemon=True).start()

    # --- SSE ---

    def stream(self, job_id: str) -> Iterator[str]:
        """Yield server-sent events for a job until it terminates."""
        job = self.get(job_id)
        if job is None:
            yield f"event: error\ndata: {json.dumps({'error': 'unknown job'})}\n\n"
            return

        q = self.subscribe(job_id)
        try:
            # Prime the connection with current state so late subscribers catch up.
            yield _sse({"type": "state", "job": job.to_dict()})
            if job.status in {"done", "error", "cancelled"}:
                yield _sse({"type": "end"})
                return
            while True:
                try:
                    event = q.get(timeout=15)
                except queue.Empty:
                    yield ": keep-alive\n\n"   # keeps proxies from closing the stream
                    continue
                if event is None:
                    return
                yield _sse(event)
                if event.get("type") == "end":
                    return
        finally:
            self.unsubscribe(job_id, q)


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, default=str)}\n\n"


store = JobStore()


class StageReporter:
    """Small helper handed to pipeline steps so they can report progress."""

    def __init__(self, job: Job, stage: str) -> None:
        self.job = job
        self.stage = stage

    def __enter__(self) -> "StageReporter":
        store.update(self.job.id, stage=self.stage, stage_progress=0.0)
        return self

    def __exit__(self, *exc: Any) -> None:
        if exc[0] is None:
            store.update(self.job.id, stage=self.stage, stage_progress=1.0)

    def progress(self, fraction: float, message: str | None = None) -> None:
        fields: dict[str, Any] = {"stage_progress": max(0.0, min(1.0, fraction))}
        if message:
            fields["message"] = message
        store.update(self.job.id, **fields)

    def log(self, message: str) -> None:
        store.log(self.job.id, message)

    def check_cancelled(self) -> None:
        if self.job.cancelled:
            raise JobCancelled(f"job {self.job.id} cancelled")


class JobCancelled(Exception):
    """Raised inside a worker thread when the client cancels the job."""


__all__ = ["Job", "JobStore", "JobCancelled", "StageReporter", "STAGES", "STAGE_KEYS", "store", "config"]
