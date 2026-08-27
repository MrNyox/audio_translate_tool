import json
import logging
import threading
from pathlib import Path

import config
from stage_one import job_store, media
from stage_two import captions, translator

logger = logging.getLogger(__name__)


def _load_ts_transcript_segments(job_dir: Path) -> list:
    ts_transcript_path = job_dir / config.TS_TRANSCRIPT_FILENAME
    if not ts_transcript_path.is_file():
        return []

    try:
        data = json.loads(ts_transcript_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read %s: %s", ts_transcript_path, exc)
        return []

    segments = data.get("segments") if isinstance(data, dict) else None
    return segments or []


def _render_subtitled_video(job_id: str, job_dir: Path, outputs: dict, translated_segments: list) -> None:
    video_path_value = outputs.get("video_path")
    if not video_path_value or not Path(video_path_value).is_file():
        logger.info(
            "Job %s: no muted video output available; skipping subtitle burn-in.",
            job_id,
        )
        return

    video_path = Path(video_path_value)

    try:
        probe = media.probe_media(video_path)
        width, height = captions.video_dimensions(probe)

        ass_content = captions.segments_to_ass(translated_segments, width, height)
        ass_path = job_dir / "captions.ass"
        ass_path.write_text(ass_content, encoding="utf-8")

        subtitled_path = job_dir / f"{config.SUBTITLED_VIDEO_PREFIX}{video_path.suffix.lower()}"

        # --- Locate the original uploaded source file (which contains the audio) ---
        audio_source_path = None
        for candidate in sorted(job_dir.glob(f"{config.SOURCE_PREFIX}.*")):
            if candidate.is_file():
                audio_source_path = candidate
                break
        # ---------------------------------------------------------------------------

        # Pass audio_source_path to burn_subtitles so it muxes the original audio back in
        captions.burn_subtitles(
            video_path,
            ass_path,
            subtitled_path,
            audio_source_path=audio_source_path
        )

        outputs["subtitled_video_path"] = str(subtitled_path)

    except media.MediaError as exc:
        # Subtitle burn-in is additive -- never fail the whole translation
        # job just because rendering the captioned video didn't work.
        logger.warning("Job %s: subtitle burn-in skipped: %s", job_id, exc)


def process_translation(job_id: str, target_language: str) -> None:
    try:
        job_store.update_job(
            job_id, status="translating", step="Translating transcript", progress=40
        )

        job_dir = config.JOB_OUTPUT_ROOT / job_id
        transcript_path = job_dir / config.TRANSCRIPT_FILENAME

        if not transcript_path.is_file():
            raise FileNotFoundError(f"Transcript not found: {transcript_path}")

        text = transcript_path.read_text(encoding="utf-8")
        if not text.strip():
            raise ValueError("Transcript is empty — nothing to translate.")

        # --- Original flat translation: unchanged behavior/output --------
        translated_text = translator.translate_text(text, target_language)

        translated_path = job_dir / config.TRANSLATED_FILENAME
        translated_path.write_text(translated_text, encoding="utf-8")

        job = job_store.get_job(job_id)
        outputs = job.get("outputs", {})
        outputs["translated_path"] = str(translated_path)

        # --- New, additive: timestamped translation + subtitles ----------
        segments = _load_ts_transcript_segments(job_dir)

        if segments:
            job_store.update_job(
                job_id, step="Translating timestamped segments", progress=65
            )

            translated_segments = translator.translate_segments(segments, target_language)

            ts_translated_path = job_dir / config.TS_TRANSLATED_FILENAME
            ts_translated_path.write_text(
                json.dumps({"segments": translated_segments}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            outputs["ts_translated_path"] = str(ts_translated_path)

            job_store.update_job(job_id, step="Rendering subtitles", progress=85)
            _render_subtitled_video(job_id, job_dir, outputs, translated_segments)

        else:
            logger.info(
                "Job %s: no timestamped segments available (forced aligner "
                "unavailable during Stage One), skipping ts_translated.json "
                "and subtitle burn-in.",
                job_id,
            )

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
