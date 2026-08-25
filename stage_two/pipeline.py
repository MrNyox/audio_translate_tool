import logging
import threading

import config
from stage_one import job_store
from stage_two import translator

logger = logging.getLogger(__name__)


def process_translation(job_id: str, target_language: str) -> None:
    try:
        job_store.update_job(
            job_id, status="translating", step="Translating transcript", progress=50
        )

        job_dir = config.JOB_OUTPUT_ROOT / job_id
        transcript_path = job_dir / config.TRANSCRIPT_FILENAME

        if not transcript_path.is_file():
            raise FileNotFoundError(f"Transcript not found: {transcript_path}")

        text = transcript_path.read_text(encoding="utf-8")
        if not text.strip():
            raise ValueError("Transcript is empty — nothing to translate.")

        translated_text = translator.translate_text(text, target_language)

        translated_path = job_dir / config.TRANSLATED_FILENAME
        translated_path.write_text(translated_text, encoding="utf-8")

        job = job_store.get_job(job_id)
        outputs = job.get("outputs", {})
        outputs["translated_path"] = str(translated_path)

        job_store.update_job(
            job_id,
            status="completed",
            step="Completed",
            progress=100,
            outputs=outputs,
        )
    except Exception as e:
        logger.exception("Translation job %s failed", job_id)
        job_store.update_job(job_id, status="failed", error=str(e))
    finally:
        job_store.clear_active_job()  # ← CRITICAL: release the active-job slot


def start_translation_job(job_id: str, target_language: str):
    worker = threading.Thread(
        target=process_translation,
        args=(job_id, target_language),
        daemon=True,
        name=f"translation-{job_id}",
    )
    worker.start()
