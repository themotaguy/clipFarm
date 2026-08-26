"""REST API + SSE progress stream."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Any

import requests
from flask import Blueprint, Response, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

import config
from app.core import ingest, media, pipeline, transcribe, vectorstore
from app.jobs import STAGES, store

bp = Blueprint("api", __name__)


# --- UI ---

@bp.get("/")
def index():
    return render_template("index.html")


# --- health ---

@bp.get("/api/health")
def health():
    checks: dict[str, Any] = {}

    checks["ffmpeg"] = {
        "ok": media.has_ffmpeg(),
        "detail": "found on PATH" if media.has_ffmpeg() else "install with `brew install ffmpeg`",
    }

    try:
        resp = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=3)
        resp.raise_for_status()
        names = {m["name"] for m in resp.json().get("models", [])}
        # Ollama reports "llama3.1:8b"; a bare "llama3.1" request resolves to :latest.
        have = lambda want: any(n == want or n.startswith(f"{want}:") for n in names)  # noqa: E731
        missing = [m for m in (config.OLLAMA_MODEL, config.OLLAMA_EMBED_MODEL) if not have(m)]
        checks["ollama"] = {
            "ok": not missing,
            "detail": f"missing model(s): {', '.join(missing)}. Run `ollama pull <model>`."
                      if missing else f"{len(names)} model(s) available",
            "models": sorted(names),
        }
    except Exception as exc:  # noqa: BLE001
        checks["ollama"] = {
            "ok": False,
            "detail": f"cannot reach {config.OLLAMA_BASE_URL} ({exc}). Run `ollama serve`.",
        }

    try:
        vectorstore.get_client().heartbeat()
        checks["chromadb"] = {"ok": True, "detail": f"persisted at {config.CHROMA_DIR}"}
    except Exception as exc:  # noqa: BLE001
        checks["chromadb"] = {"ok": False, "detail": str(exc)}

    checks["whisper"] = {
        "ok": True,
        "detail": f"{config.WHISPER_MODEL} on {config.WHISPER_DEVICE} "
                  f"({config.WHISPER_COMPUTE_TYPE}) — downloads on first run",
    }

    return jsonify({
        "ok": all(c["ok"] for c in checks.values()),
        "checks": checks,
        "config": {
            "whisper_model": config.WHISPER_MODEL,
            "ollama_model": config.OLLAMA_MODEL,
            "embed_model": config.OLLAMA_EMBED_MODEL,
            "max_clips": config.MAX_CLIPS,
            "clip_seconds": [config.CLIP_MIN_SECONDS, config.CLIP_MAX_SECONDS],
            "resolution": f"{config.RENDER_WIDTH}x{config.RENDER_HEIGHT}",
            "burn_captions": config.BURN_CAPTIONS,
        },
        "stages": [{"key": k, "label": lbl} for k, lbl in STAGES],
    })


# --- jobs ---

@bp.post("/api/jobs")
def create_job():
    """Start a job from an uploaded file (multipart `file`) or a JSON `{"url": ...}`."""
    upload = request.files.get("file")

    if upload and upload.filename:
        filename = secure_filename(upload.filename) or "upload"
        suffix = Path(filename).suffix.lower()
        if suffix not in config.ALLOWED_EXTENSIONS:
            return jsonify({
                "error": f"Unsupported file type `{suffix or 'unknown'}`. "
                         f"Allowed: {', '.join(sorted(config.ALLOWED_EXTENSIONS))}"
            }), 400

        job = store.create("upload", filename)
        dest_dir = config.UPLOAD_DIR / job.id
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"source{suffix}"
        upload.save(dest)
        if dest.stat().st_size == 0:
            store.delete(job.id)
            return jsonify({"error": "Uploaded file is empty."}), 400
        job.media["input_path"] = str(dest)

    else:
        payload = request.get_json(silent=True) or request.form
        url = (payload.get("url") or "").strip()
        if not url:
            return jsonify({"error": "Provide a `file` upload or a `url` field."}), 400
        if not ingest.is_url(url):
            return jsonify({"error": "`url` must start with http:// or https://"}), 400
        job = store.create("url", url)

    store.run(job, pipeline.process)
    return jsonify(job.to_dict()), 202


@bp.get("/api/jobs")
def list_jobs():
    return jsonify({"jobs": [j.to_dict() for j in store.list()]})


@bp.get("/api/jobs/<job_id>")
def get_job(job_id: str):
    job = store.get(job_id)
    if job is None:
        return jsonify({"error": "unknown job"}), 404
    return jsonify(job.to_dict())


@bp.get("/api/jobs/<job_id>/events")
def job_events(job_id: str):
    """Server-sent events: one `state` message per pipeline update, plus logs."""
    if store.get(job_id) is None:
        return jsonify({"error": "unknown job"}), 404
    return Response(
        store.stream(job_id),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@bp.post("/api/jobs/<job_id>/cancel")
def cancel_job(job_id: str):
    if store.get(job_id) is None:
        return jsonify({"error": "unknown job"}), 404
    return jsonify({"cancelled": store.cancel(job_id)})


@bp.delete("/api/jobs/<job_id>")
def delete_job(job_id: str):
    job = store.get(job_id)
    if job is None:
        return jsonify({"error": "unknown job"}), 404

    store.delete(job_id)
    for directory in (config.UPLOAD_DIR / job_id, config.MEDIA_DIR / job_id,
                      config.CLIP_DIR / job_id):
        shutil.rmtree(directory, ignore_errors=True)
    pipeline.result_path(job_id).unlink(missing_ok=True)
    try:
        vectorstore.get_client().delete_collection(f"job_{job_id}")
    except Exception:  # noqa: BLE001 - collection may never have been created
        pass
    return jsonify({"deleted": True})


# --- transcript & retrieval ---

@bp.get("/api/jobs/<job_id>/transcript")
def get_transcript(job_id: str):
    job = store.get(job_id)
    if job is None:
        return jsonify({"error": "unknown job"}), 404
    if request.args.get("format") == "srt":
        return Response(
            transcribe.to_srt(job.transcript),
            mimetype="text/plain",
            headers={"Content-Disposition": f'attachment; filename="{job_id}.srt"'},
        )
    return jsonify({
        "job_id": job_id,
        "segments": job.transcript,
        "text": transcribe.full_text(job.transcript),
    })


@bp.post("/api/jobs/<job_id>/search")
def search(job_id: str):
    """Semantic search over the job's ChromaDB collection."""
    if store.get(job_id) is None:
        return jsonify({"error": "unknown job"}), 404

    payload = request.get_json(silent=True) or {}
    query = (payload.get("query") or request.args.get("q") or "").strip()
    if not query:
        return jsonify({"error": "`query` is required"}), 400
    k = max(1, min(int(payload.get("k", 8)), 50))

    try:
        hits = vectorstore.search_job(job_id, query, k=k)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"index unavailable: {exc}"}), 409
    return jsonify({"query": query, "hits": hits})


# --- media files ---

def _find_clip(job_id: str, clip_id: str) -> dict[str, Any] | None:
    job = store.get(job_id)
    if job is None:
        return None
    return next((c for c in job.clips if c["id"] == clip_id), None)


def _serve(path_str: str | None, mimetype: str, *, download_name: str | None = None):
    if not path_str:
        return jsonify({"error": "file not available"}), 404
    path = Path(path_str)
    if not path.exists():
        return jsonify({"error": "file not found on disk"}), 404
    return send_file(
        path,
        mimetype=mimetype,
        conditional=True,          # honour Range requests so <video> can seek
        download_name=download_name,
        as_attachment=bool(download_name),
    )


@bp.get("/api/jobs/<job_id>/clips/<clip_id>/file")
def clip_file(job_id: str, clip_id: str):
    clip = _find_clip(job_id, clip_id)
    if clip is None:
        return jsonify({"error": "unknown clip"}), 404
    name = None
    if request.args.get("download") == "1":
        name = f"{_slug(clip['title'])}-{clip_id}.mp4"
    return _serve(clip.get("path"), "video/mp4", download_name=name)


@bp.get("/api/jobs/<job_id>/clips/<clip_id>/thumbnail")
def clip_thumbnail(job_id: str, clip_id: str):
    clip = _find_clip(job_id, clip_id)
    if clip is None:
        return jsonify({"error": "unknown clip"}), 404
    return _serve(clip.get("thumbnail"), "image/jpeg")


@bp.get("/api/jobs/<job_id>/clips/<clip_id>/srt")
def clip_srt(job_id: str, clip_id: str):
    clip = _find_clip(job_id, clip_id)
    if clip is None:
        return jsonify({"error": "unknown clip"}), 404
    return _serve(clip.get("subtitles_srt"), "text/plain",
                  download_name=f"{_slug(clip['title'])}-{clip_id}.srt")


@bp.get("/api/jobs/<job_id>/poster")
def job_poster(job_id: str):
    job = store.get(job_id)
    if job is None:
        return jsonify({"error": "unknown job"}), 404
    return _serve(job.media.get("poster"), "image/jpeg")


def _slug(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_ " else "" for ch in text or "")
    slug = "-".join(cleaned.split()).lower()[:60].strip("-")
    return slug or uuid.uuid4().hex[:8]
