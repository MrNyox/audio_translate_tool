import threading
import uuid
from datetime import datetime, timezone

_JOBS = {}
_ACTIVE_JOB_ID = None
_LOCK = threading.RLock()


def _copy_job(job):
    copied = dict(job)
    copied["outputs"] = dict(job.get("outputs") or {})
    return copied


def create_job(original_filename: str) -> str:
    job_id = uuid.uuid4().hex[:12]
    now = datetime.now(timezone.utc).isoformat()

    job = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0,
        "step": "Queued",
        "error": None,
        "original_filename": original_filename,
        "media_kind": None,
        "created_at": now,
        "outputs": {
            "audio_path": None,
            "transcript_path": None,
            "video_path": None,
        },
    }

    with _LOCK:
        _JOBS[job_id] = job

    return job_id


def get_job(job_id: str):
    with _LOCK:
        job = _JOBS.get(job_id)
        if not job:
            return None
        return _copy_job(job)


def update_job(job_id: str, **fields):
    with _LOCK:
        job = _JOBS.get(job_id)
        if not job:
            return None

        for key, value in fields.items():
            if key == "outputs":
                if isinstance(value, dict):
                    job["outputs"].update(value)
            else:
                job[key] = value

        return _copy_job(job)


def delete_job(job_id: str) -> None:
    global _ACTIVE_JOB_ID

    with _LOCK:
        _JOBS.pop(job_id, None)
        if _ACTIVE_JOB_ID == job_id:
            _ACTIVE_JOB_ID = None


def get_active_job_id():
    with _LOCK:
        return _ACTIVE_JOB_ID


def set_active_job(job_id: str) -> None:
    global _ACTIVE_JOB_ID

    with _LOCK:
        _ACTIVE_JOB_ID = job_id


def clear_active_job() -> None:
    global _ACTIVE_JOB_ID

    with _LOCK:
        _ACTIVE_JOB_ID = None
