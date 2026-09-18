"""Job registry, progress math and the SSE pub/sub bus."""
from __future__ import annotations

import queue
import threading
import time

import pytest

from app.jobs import STAGE_KEYS, Job, JobCancelled, JobStore, StageReporter


@pytest.fixture
def jobstore():
    return JobStore()


# --- progress ---

def test_overall_progress_spans_stages_evenly():
    job = Job(id="a", source="upload", source_label="x")
    job.stage = STAGE_KEYS[0]
    job.stage_progress = 0.0
    assert job.overall_progress() == 0.0

    job.stage = STAGE_KEYS[0]
    job.stage_progress = 1.0
    assert abs(job.overall_progress() - 1 / len(STAGE_KEYS)) < 1e-9

    job.stage = STAGE_KEYS[-1]
    job.stage_progress = 1.0
    assert job.overall_progress() == 1.0


def test_overall_progress_is_one_when_done():
    job = Job(id="a", source="upload", source_label="x", status="done")
    job.stage = STAGE_KEYS[1]
    assert job.overall_progress() == 1.0


def test_overall_progress_clamps_out_of_range_stage_progress():
    job = Job(id="a", source="upload", source_label="x")
    job.stage = STAGE_KEYS[2]
    job.stage_progress = 5.0
    assert job.overall_progress() <= 1.0


def test_overall_progress_unknown_stage_is_zero():
    job = Job(id="a", source="upload", source_label="x")
    job.stage = "not-a-stage"
    assert job.overall_progress() == 0.0


def test_to_dict_omits_transcript_unless_asked():
    job = Job(id="a", source="upload", source_label="x")
    job.transcript = [{"start": 0, "end": 1, "text": "hi"}]
    assert "transcript" not in job.to_dict()
    assert job.to_dict(include_transcript=True)["transcript"] == job.transcript


# --- registry ---

def test_create_get_list_delete(jobstore):
    job = jobstore.create("upload", "a.mp4")
    assert jobstore.get(job.id) is job
    assert jobstore.list() == [job]
    assert jobstore.delete(job.id) is True
    assert jobstore.get(job.id) is None
    assert jobstore.delete(job.id) is False


def test_list_is_newest_first(jobstore):
    a = jobstore.create("upload", "a.mp4")
    time.sleep(0.01)
    b = jobstore.create("upload", "b.mp4")
    assert [j.id for j in jobstore.list()] == [b.id, a.id]


def test_update_ignores_unknown_fields(jobstore):
    job = jobstore.create("upload", "a.mp4")
    jobstore.update(job.id, message="hello", not_a_field=123)
    assert job.message == "hello"
    assert not hasattr(job, "not_a_field")


def test_update_unknown_job_returns_none(jobstore):
    assert jobstore.update("missing", message="x") is None


def test_cancel_sets_flag_and_is_idempotent_on_terminal(jobstore):
    job = jobstore.create("upload", "a.mp4")
    assert jobstore.cancel(job.id) is True
    assert job.cancelled is True

    job.status = "done"
    assert jobstore.cancel(job.id) is False
    assert jobstore.cancel("missing") is False


def test_delete_cancels_a_running_job(jobstore):
    job = jobstore.create("upload", "a.mp4")
    jobstore.delete(job.id)
    assert job.cancelled is True


# --- pub/sub ---

def test_subscribers_receive_published_state(jobstore):
    job = jobstore.create("upload", "a.mp4")
    q = jobstore.subscribe(job.id)
    jobstore.update(job.id, message="step one")
    event = q.get(timeout=1)
    assert event["type"] == "state"
    assert event["job"]["message"] == "step one"


def test_log_events_do_not_change_state(jobstore):
    job = jobstore.create("upload", "a.mp4")
    q = jobstore.subscribe(job.id)
    jobstore.log(job.id, "something happened")
    event = q.get(timeout=1)
    assert event["type"] == "log"
    assert event["message"] == "something happened"


def test_unsubscribe_stops_delivery(jobstore):
    job = jobstore.create("upload", "a.mp4")
    q = jobstore.subscribe(job.id)
    jobstore.unsubscribe(job.id, q)
    jobstore.update(job.id, message="after")
    with pytest.raises(queue.Empty):
        q.get(timeout=0.1)


def test_full_subscriber_queue_never_blocks_publisher(jobstore):
    job = jobstore.create("upload", "a.mp4")
    q = jobstore.subscribe(job.id)
    for _ in range(q.maxsize + 50):
        jobstore.update(job.id, message="flood")   # must not raise or hang
    assert q.qsize() == q.maxsize


def test_delete_wakes_streaming_subscribers(jobstore):
    job = jobstore.create("upload", "a.mp4")
    q = jobstore.subscribe(job.id)
    jobstore.delete(job.id)
    assert q.get(timeout=1) is None


# --- execution ---

def test_run_marks_job_done_and_emits_end(jobstore):
    job = jobstore.create("upload", "a.mp4")
    q = jobstore.subscribe(job.id)
    finished = threading.Event()

    def target(j):
        j.clips = [{"id": "clip-01"}]
        finished.set()

    jobstore.run(job, target)
    assert finished.wait(5)

    types = []
    deadline = time.time() + 5
    while time.time() < deadline:
        try:
            types.append(q.get(timeout=0.5)["type"])
        except queue.Empty:
            break
        if types[-1] == "end":
            break
    assert "end" in types
    assert job.status == "done"
    assert "1 clip(s)" in job.message


def test_run_records_error_from_worker(jobstore):
    job = jobstore.create("upload", "a.mp4")

    def target(_j):
        raise ValueError("boom")

    jobstore.run(job, target)
    for _ in range(100):
        if job.status == "error":
            break
        time.sleep(0.05)
    assert job.status == "error"
    assert "ValueError: boom" in job.error


def test_run_records_cancellation(jobstore):
    job = jobstore.create("upload", "a.mp4")

    def target(_j):
        raise JobCancelled("stopped")

    jobstore.run(job, target)
    for _ in range(100):
        if job.status == "cancelled":
            break
        time.sleep(0.05)
    assert job.status == "cancelled"
    assert job.error is None


# --- StageReporter ---

def test_stage_reporter_sets_and_completes_stage(monkeypatch):
    from app import jobs as jobs_mod

    local = JobStore()
    monkeypatch.setattr(jobs_mod, "store", local)
    job = local.create("upload", "a.mp4")

    with StageReporter(job, "transcribe") as rep:
        assert job.stage == "transcribe"
        rep.progress(0.5, "halfway")
        assert job.stage_progress == 0.5
        assert job.message == "halfway"
    assert job.stage_progress == 1.0


def test_stage_reporter_leaves_progress_alone_on_error(monkeypatch):
    from app import jobs as jobs_mod

    local = JobStore()
    monkeypatch.setattr(jobs_mod, "store", local)
    job = local.create("upload", "a.mp4")

    with pytest.raises(RuntimeError):
        with StageReporter(job, "render") as rep:
            rep.progress(0.25)
            raise RuntimeError("render blew up")
    assert job.stage_progress == 0.25


def test_stage_reporter_check_cancelled_raises(monkeypatch):
    from app import jobs as jobs_mod

    local = JobStore()
    monkeypatch.setattr(jobs_mod, "store", local)
    job = local.create("upload", "a.mp4")
    rep = StageReporter(job, "ingest")
    rep.check_cancelled()          # no-op while running
    local.cancel(job.id)
    with pytest.raises(JobCancelled):
        rep.check_cancelled()


def test_stage_reporter_clamps_progress(monkeypatch):
    from app import jobs as jobs_mod

    local = JobStore()
    monkeypatch.setattr(jobs_mod, "store", local)
    job = local.create("upload", "a.mp4")
    rep = StageReporter(job, "score")
    rep.progress(-3.0)
    assert job.stage_progress == 0.0
    rep.progress(9.0)
    assert job.stage_progress == 1.0
