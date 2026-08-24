import mimetypes
import re
import threading
from pathlib import Path

from flask import Blueprint, jsonify, request, send_file
from werkzeug.utils import secure_filename

import config
from stage_one import asr, job_store, media, pipeline

bp = Blueprint("api", __name__)

JOB_ID_RE = re.compile(r"^[a-f0-9]{12}$")

RUNNING_STATUSES = {
    "queued",
    "loading_model",
    "splitting_media",
    "transcribing",
    "saving_outputs",
}


def _ok(data=None, status=200):
    return jsonify({"ok": True, "data": data or {}}), status


def _err(message, status=400):
    return jsonify({"ok": False, "error": message}), status


def _is_path_inside(base: Path, target: Path) -> bool:
    try:
        return target.is_relative_to(base)
    except AttributeError:
        try:
            target.relative_to(base)
            return True
        except ValueError:
            return False


def _public_job(job):
    if not job:
        return {}

    job_id = job["job_id"]
    status = job["status"]
    outputs = job.get("outputs") or {}

    def output_url(kind):
        path_value = outputs.get(f"{kind}_path")
        if status != "completed" or not path_value:
            return None
        if not Path(path_value).is_file():
            return None
        return f"/api/jobs/{job_id}/outputs/{kind}"

    return {
        "job_id": job_id,
        "status": status,
        "progress": job.get("progress", 0),
        "step": job.get("step", ""),
        "error": job.get("error"),
        "media_kind": job.get("media_kind"),
        "created_at": job.get("created_at"),
        "outputs": {
            "audio_url": output_url("audio"),
            "transcript_url": output_url("transcript"),
            "video_url": output_url("video"),
        },
    }


def _safe_job_dir(job_id: str):
    if not JOB_ID_RE.fullmatch(job_id):
        return None

    root = config.JOB_OUTPUT_ROOT.resolve()
    candidate = (root / job_id).resolve()

    if not candidate.is_dir():
        return None

    if not _is_path_inside(root, candidate):
        return None

    return candidate


def _send_exact_file(job_id: str, filename: str, mimetype: str = None):
    job_dir = _safe_job_dir(job_id)
    if not job_dir:
        return _err("Job not found.", 404)

    path = (job_dir / filename).resolve()

    if not path.is_file() or not _is_path_inside(job_dir, path):
        return _err("Output file not found.", 404)

    resolved_mime = mimetype or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return send_file(
        path,
        as_attachment=True,
        download_name=path.name,
        mimetype=resolved_mime,
    )


def _send_glob_file(job_id: str, prefix: str, mimetype: str = None):
    job_dir = _safe_job_dir(job_id)
    if not job_dir:
        return _err("Job not found.", 404)

    candidates = sorted(job_dir.glob(f"{prefix}.*"))
    path = None

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file() and _is_path_inside(job_dir, resolved):
            path = resolved
            break

    if path is None:
        return _err("Output file not found.", 404)

    resolved_mime = mimetype or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return send_file(
        path,
        as_attachment=True,
        download_name=path.name,
        mimetype=resolved_mime,
    )


@bp.get("/api/health")
def health():
    return _ok(
        {
            "status": "online",
            "model_loaded": asr.is_model_loaded(),
            "ffmpeg_available": media.ffmpeg_available(),
        }
    )


@bp.post("/api/jobs")
def create_job():
    active_job_id = job_store.get_active_job_id()

    if active_job_id:
        active_job = job_store.get_job(active_job_id)
        if active_job and active_job.get("status") in RUNNING_STATUSES:
            return _err(
                "Another job is already processing. Please wait for it to finish.",
                409,
            )
        job_store.clear_active_job()

    if not request.files:
        return _err("No file uploaded.", 400)

    uploaded = next(iter(request.files.values()), None)

    if uploaded is None or not uploaded.filename:
        return _err("No file selected.", 400)

    original_filename = uploaded.filename
    extension = Path(original_filename).suffix.lstrip(".").lower()

    if not extension or extension not in config.ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(config.ALLOWED_EXTENSIONS))
        return _err(f"Unsupported file type. Allowed extensions: {allowed}.", 400)

    safe_display_name = secure_filename(original_filename) or f"upload.{extension}"

    job_id = job_store.create_job(safe_display_name)
    job_dir = config.JOB_OUTPUT_ROOT / job_id

    try:
        job_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        job_store.delete_job(job_id)
        return _err("Could not create the job folder.", 500)

    source_path = job_dir / f"{config.SOURCE_PREFIX}.{extension}"

    try:
        uploaded.save(source_path)
    except OSError:
        job_store.delete_job(job_id)
        return _err("Could not save the uploaded file.", 500)

    job_store.update_job(
        job_id,
        status="queued",
        progress=0,
        step="Queued",
        error=None,
    )

    job_store.set_active_job(job_id)

    worker = threading.Thread(
        target=pipeline.process_job,
        args=(job_id,),
        daemon=True,
    )
    worker.start()

    job = job_store.get_job(job_id)
    return _ok(_public_job(job), 201)


@bp.get("/api/jobs/<job_id>")
def get_job(job_id):
    if not JOB_ID_RE.fullmatch(job_id):
        return _err("Job not found.", 404)

    job = job_store.get_job(job_id)
    if not job:
        return _err("Job not found.", 404)

    return _ok(_public_job(job))


@bp.get("/api/jobs/<job_id>/outputs/audio")
def download_audio(job_id):
    return _send_exact_file(
        job_id,
        config.ASR_AUDIO_FILENAME,
        mimetype="audio/wav",
    )


@bp.get("/api/jobs/<job_id>/outputs/transcript")
def download_transcript(job_id):
    return _send_exact_file(
        job_id,
        config.TRANSCRIPT_FILENAME,
        mimetype="text/plain; charset=utf-8",
    )


@bp.get("/api/jobs/<job_id>/outputs/video")
def download_video(job_id):
    return _send_glob_file(
        job_id,
        config.MUTED_VIDEO_PREFIX,
        mimetype=None,
    )
