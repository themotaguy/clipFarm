"""Flask REST surface. The pipeline itself is stubbed out."""
from __future__ import annotations

import io
import json

import pytest

import config
from app.core import pipeline
from app.jobs import store


@pytest.fixture
def client(tmp_path, monkeypatch):
    """App with isolated data dirs and a no-op pipeline."""
    for name in ("UPLOAD_DIR", "MEDIA_DIR", "CLIP_DIR", "JOB_DIR", "CHROMA_DIR"):
        d = tmp_path / name.lower()
        d.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(config, name, d)

    # Don't let create_app() adopt real jobs from the developer's data dir.
    monkeypatch.setattr(pipeline, "restore_all", lambda: 0)

    def fake_process(job):
        store.update(job.id, clips=[{"id": "clip-01", "title": "Stub", "path": None}])

    monkeypatch.setattr(pipeline, "process", fake_process)

    from app import create_app

    app = create_app()
    app.config.update(TESTING=True)
    with app.test_client() as c:
        yield c

    for job in list(store.list()):
        store.delete(job.id)


# --- health ---

def test_health_reports_every_dependency(client, monkeypatch):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.get_json()
    assert set(body["checks"]) == {
        "ffmpeg", "ollama", "chromadb", "whisper", "captions",
    }
    assert [s["key"] for s in body["stages"]] == [
        "ingest", "transcribe", "index", "retrieve", "score", "render",
    ]
    assert body["config"]["ollama_model"] == config.OLLAMA_MODEL


def test_health_flags_missing_ffmpeg(client, monkeypatch):
    from app.core import media

    monkeypatch.setattr(media, "has_ffmpeg", lambda: False)
    body = client.get("/api/health").get_json()
    assert body["checks"]["ffmpeg"]["ok"] is False
    assert body["ok"] is False


def test_health_reports_ffmpeg_without_libass(client, monkeypatch):
    from app.core import media

    monkeypatch.setattr(media, "has_ffmpeg", lambda: True)
    monkeypatch.setattr(media, "has_filter", lambda name: False)
    check = client.get("/api/health").get_json()["checks"]["ffmpeg"]
    # ffmpeg is still usable, so this must not fail the overall health check.
    assert check["ok"] is True
    assert check["has_libass"] is False
    assert "libass" in check["detail"]


def test_health_reports_the_pillow_caption_fallback(client, monkeypatch):
    from app.core import media, pngcaptions

    monkeypatch.setattr(media, "has_filter", lambda name: False)
    monkeypatch.setattr(pngcaptions, "available", lambda: True)
    check = client.get("/api/health").get_json()["checks"]["captions"]
    assert check["ok"] is True
    assert check["backend"] == "pillow"


def test_health_fails_when_no_caption_renderer_exists(client, monkeypatch):
    from app.core import media, pngcaptions

    monkeypatch.setattr(media, "has_filter", lambda name: False)
    monkeypatch.setattr(pngcaptions, "available", lambda: False)
    body = client.get("/api/health").get_json()
    assert body["checks"]["captions"]["ok"] is False
    assert body["ok"] is False


def test_health_caption_check_passes_when_captions_are_off(client, monkeypatch):
    monkeypatch.setattr(config, "BURN_CAPTIONS", False)
    check = client.get("/api/health").get_json()["checks"]["captions"]
    assert check["ok"] is True
    assert check["backend"] == "none"


# --- job creation ---

def test_create_job_from_upload(client):
    data = {"file": (io.BytesIO(b"fake mp4 bytes"), "talk.mp4")}
    resp = client.post("/api/jobs", data=data, content_type="multipart/form-data")
    assert resp.status_code == 202
    body = resp.get_json()
    assert body["source"] == "upload"
    assert body["source_label"] == "talk.mp4"
    assert body["id"]


def test_create_job_rejects_unsupported_extension(client):
    data = {"file": (io.BytesIO(b"nope"), "notes.txt")}
    resp = client.post("/api/jobs", data=data, content_type="multipart/form-data")
    assert resp.status_code == 400
    assert "Unsupported file type" in resp.get_json()["error"]


def test_create_job_rejects_empty_upload(client):
    data = {"file": (io.BytesIO(b""), "empty.mp4")}
    resp = client.post("/api/jobs", data=data, content_type="multipart/form-data")
    assert resp.status_code == 400
    assert "empty" in resp.get_json()["error"].lower()


def test_create_job_from_url(client):
    resp = client.post("/api/jobs", json={"url": "https://example.com/video"})
    assert resp.status_code == 202
    assert resp.get_json()["source"] == "url"


def test_create_job_rejects_non_http_url(client):
    resp = client.post("/api/jobs", json={"url": "file:///etc/passwd"})
    assert resp.status_code == 400
    assert "http" in resp.get_json()["error"]


def test_create_job_requires_a_source(client):
    resp = client.post("/api/jobs", json={})
    assert resp.status_code == 400


# --- job lifecycle ---

def test_unknown_job_returns_404(client):
    for path in (
        "/api/jobs/nope",
        "/api/jobs/nope/transcript",
        "/api/jobs/nope/events",
        "/api/jobs/nope/poster",
        "/api/jobs/nope/clips/clip-01/file",
        "/api/jobs/nope/clips/clip-01/thumbnail",
        "/api/jobs/nope/clips/clip-01/srt",
    ):
        assert client.get(path).status_code == 404, path
    assert client.post("/api/jobs/nope/cancel").status_code == 404
    assert client.delete("/api/jobs/nope").status_code == 404
    assert client.post("/api/jobs/nope/search", json={"query": "x"}).status_code == 404


def test_list_jobs_includes_created_job(client):
    client.post("/api/jobs", json={"url": "https://example.com/a"})
    body = client.get("/api/jobs").get_json()
    assert len(body["jobs"]) == 1


def test_delete_job_removes_it(client):
    job_id = client.post("/api/jobs", json={"url": "https://example.com/a"}).get_json()["id"]
    assert client.delete(f"/api/jobs/{job_id}").get_json()["deleted"] is True
    assert client.get(f"/api/jobs/{job_id}").status_code == 404


def test_search_requires_a_query(client):
    job_id = client.post("/api/jobs", json={"url": "https://example.com/a"}).get_json()["id"]
    resp = client.post(f"/api/jobs/{job_id}/search", json={})
    assert resp.status_code == 400
    assert "query" in resp.get_json()["error"]


def test_search_returns_409_when_index_missing(client):
    job_id = client.post("/api/jobs", json={"url": "https://example.com/a"}).get_json()["id"]
    resp = client.post(f"/api/jobs/{job_id}/search", json={"query": "anything"})
    # No collection was ever created for this stubbed job.
    assert resp.status_code == 409


def test_transcript_srt_format(client, transcript, monkeypatch):
    job_id = client.post("/api/jobs", json={"url": "https://example.com/a"}).get_json()["id"]
    store.update(job_id, transcript=transcript)
    resp = client.get(f"/api/jobs/{job_id}/transcript?format=srt")
    assert resp.status_code == 200
    assert "-->" in resp.get_data(as_text=True)
    assert "attachment" in resp.headers["Content-Disposition"]


def test_clip_file_404_when_path_missing(client):
    job_id = client.post("/api/jobs", json={"url": "https://example.com/a"}).get_json()["id"]
    store.update(job_id, clips=[{"id": "clip-01", "title": "T", "path": None}])
    assert client.get(f"/api/jobs/{job_id}/clips/clip-01/file").status_code == 404


def test_clip_file_served_when_present(client, tmp_path):
    job_id = client.post("/api/jobs", json={"url": "https://example.com/a"}).get_json()["id"]
    video = tmp_path / "clip-01.mp4"
    video.write_bytes(b"\x00" * 2048)
    store.update(job_id, clips=[{"id": "clip-01", "title": "T", "path": str(video)}])

    resp = client.get(f"/api/jobs/{job_id}/clips/clip-01/file")
    assert resp.status_code == 200
    assert resp.headers["Content-Type"] == "video/mp4"


def test_ui_index_renders(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"<html" in resp.get_data().lower()


# --- SSE ---

def test_events_stream_primes_and_ends_for_finished_job(client):
    job_id = client.post("/api/jobs", json={"url": "https://example.com/a"}).get_json()["id"]
    store.update(job_id, status="done")

    resp = client.get(f"/api/jobs/{job_id}/events")
    assert resp.status_code == 200
    assert resp.headers["Content-Type"].startswith("text/event-stream")
    payload = resp.get_data(as_text=True)
    events = [json.loads(l[len("data: "):]) for l in payload.splitlines()
              if l.startswith("data: ")]
    assert events[0]["type"] == "state"
    assert events[-1]["type"] == "end"


def test_search_rejects_non_numeric_k(client):
    job_id = client.post("/api/jobs", json={"url": "https://example.com/a"}).get_json()["id"]
    resp = client.post(f"/api/jobs/{job_id}/search", json={"query": "x", "k": "abc"})
    assert resp.status_code == 400
    assert "k" in resp.get_json()["error"]


def test_search_clamps_out_of_range_k(client):
    job_id = client.post("/api/jobs", json={"url": "https://example.com/a"}).get_json()["id"]
    # Negative and oversized k are clamped, not rejected; the 409 here means the
    # request got as far as looking for the (nonexistent) collection.
    for k in (-5, 10_000):
        resp = client.post(f"/api/jobs/{job_id}/search", json={"query": "x", "k": k})
        assert resp.status_code == 409, k
