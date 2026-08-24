from pathlib import Path

import config
from stage_one import asr, job_store, media


def _job_dir(job_id: str) -> Path:
    return config.JOB_OUTPUT_ROOT / job_id


def _find_source_file(job_dir: Path) -> Path:
    candidates = sorted(job_dir.glob(f"{config.SOURCE_PREFIX}.*"))

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise RuntimeError("Uploaded source file is missing.")


def process_job(job_id: str) -> None:
    try:
        job = job_store.get_job(job_id)
        if not job:
            return

        job_dir = _job_dir(job_id)
        if not job_dir.is_dir():
            raise RuntimeError("Job folder is missing.")

        source_path = _find_source_file(job_dir)

        job_store.update_job(
            job_id,
            status="loading_model",
            step="Loading model",
            progress=10,
            error=None,
        )

        asr.load_model()

        job_store.update_job(
            job_id,
            status="splitting_media",
            step="Analyzing media",
            progress=25,
        )

        probe = media.probe_media(source_path)
        video_present = media.is_video(probe)
        audio_present = media.has_audio_stream(probe)

        media_kind = "video" if video_present else "audio"

        job_store.update_job(
            job_id,
            media_kind=media_kind,
        )

        if not audio_present:
            raise media.MediaError(
                "No audio stream found in the uploaded media file."
            )

        audio_path = job_dir / config.ASR_AUDIO_FILENAME

        job_store.update_job(
            job_id,
            status="splitting_media",
            step="Extracting audio",
            progress=30,
        )

        media.extract_audio(
            source_path,
            audio_path,
            config.ASR_SAMPLE_RATE,
        )

        video_path = None

        if video_present:
            job_store.update_job(
                job_id,
                step="Extracting muted video",
                progress=45,
            )

            video_path = job_dir / f"{config.MUTED_VIDEO_PREFIX}{source_path.suffix.lower()}"

            media.extract_muted_video(
                source_path,
                video_path,
            )

        job_store.update_job(
            job_id,
            status="transcribing",
            step="Transcribing speech",
            progress=60,
        )

        transcript_text = asr.transcribe_file(audio_path)

        job_store.update_job(
            job_id,
            status="saving_outputs",
            step="Saving outputs",
            progress=90,
        )

        transcript_path = job_dir / config.TRANSCRIPT_FILENAME
        transcript_path.write_text(
            (transcript_text or "").strip(),
            encoding="utf-8",
        )

        job_store.update_job(
            job_id,
            outputs={
                "audio_path": str(audio_path),
                "transcript_path": str(transcript_path),
                "video_path": str(video_path) if video_path else None,
            },
        )

        if not audio_path.is_file():
            raise RuntimeError("Expected audio output is missing.")

        if not transcript_path.is_file():
            raise RuntimeError("Expected transcript output is missing.")

        if video_present and (not video_path or not video_path.is_file()):
            raise RuntimeError("Expected muted video output is missing.")

        job_store.update_job(
            job_id,
            status="completed",
            step="Completed",
            progress=100,
            error=None,
        )

    except Exception as exc:
        job_store.update_job(
            job_id,
            status="failed",
            error=str(exc) or "Unknown processing error.",
        )

    finally:
        job_store.clear_active_job()
