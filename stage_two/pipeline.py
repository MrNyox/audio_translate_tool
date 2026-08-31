import json
import logging
import threading
import re
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

        # translated_segments was already run through normalize_subtitle_pacing
        # in process_translation before this function was called, so tell
        # segments_to_ass not to normalize a second time.
        ass_content = captions.segments_to_ass(
            translated_segments, width, height, apply_pacing=False
        )
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

def _sync_flat_translation_to_segments(translated_text: str, original_segments: list) -> list:
    """
    Distributes the high-quality flat translation (from translated.txt)
    across the original timestamped segments proportionally based on
    the original text's word count.
    """
    if not translated_text or not original_segments:
        return []

    # Tokenize translated text into words (ignoring all whitespace/newlines)
    translated_words = re.findall(r'\S+', translated_text)
    if not translated_words:
        return []

    # Calculate weights of original segments (word count)
    original_weights = []
    for seg in original_segments:
        orig_text = str(seg.get("text", "")).strip()
        words = len(re.findall(r'\S+', orig_text))
        original_weights.append(max(1, words))  # avoid zero weight for silence/empty segments

    total_orig_weight = sum(original_weights)
    if total_orig_weight == 0:
        total_orig_weight = 1

    total_trans_words = len(translated_words)

    synced_segments = []
    trans_cursor = 0

    for i, seg in enumerate(original_segments):
        weight = original_weights[i]

        # Calculate how many translated words this segment should get
        target_words = (weight / total_orig_weight) * total_trans_words

        # Determine end index for this segment
        trans_end_idx = round(trans_cursor + target_words)

        # Ensure the last segment captures all remaining words
        if i == len(original_segments) - 1:
            trans_end_idx = total_trans_words

        # Safety bounds
        trans_end_idx = max(trans_cursor, min(trans_end_idx, total_trans_words))

        assigned_words = translated_words[trans_cursor:trans_end_idx]
        assigned_text = " ".join(assigned_words)

        synced_segments.append({
            "start": seg.get("start"),
            "end": seg.get("end"),
            "text": assigned_text
        })

        trans_cursor = trans_end_idx

    return synced_segments

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
                job_id, step="Mapping high-quality translation to timestamps", progress=65
            )

            # Map the high-quality flat translation to the original timestamps
            if translated_text.strip():
                translated_segments = _sync_flat_translation_to_segments(translated_text, segments)
            else:
                # Fallback to the old method if the flat translation is somehow empty
                logger.warning("Job %s: Flat translation empty, falling back to segment translation.", job_id)
                translated_segments = translator.translate_segments(segments, target_language)

            # --- Normalize subtitle pacing for comfortable reading ---
            # THIS IS THE MAGIC: It takes the newly assigned text, checks its length,
            # and automatically stretches the timestamps into the next pauses so the
            # viewer has time to read the longer/shorter translated text!
            translated_segments = captions.normalize_subtitle_pacing(translated_segments)
            # --------------------------------------------------------------

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
