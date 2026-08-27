import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Output root. Automatically created at runtime by app.py.
JOB_OUTPUT_ROOT = Path(
    os.getenv("STAGE_ONE_OUTPUT_ROOT", str(BASE_DIR / "job_output"))
).resolve()

# Qwen3-ASR model id or local folder path.
_LOCAL_ASR_DIR = BASE_DIR / "models" / "Qwen3-ASR-1.7B"
MODEL_ID = (
    os.getenv("QWEN3_ASR_MODEL")
    or os.getenv("QWEN3_ASR")  # legacy name
    or (str(_LOCAL_ASR_DIR) if _LOCAL_ASR_DIR.is_dir() else "Qwen/Qwen3-ASR-1.7B")
)

# Qwen3-ForcedAligner model id or local folder path. Loaded alongside the ASR
# model so `.transcribe(..., return_time_stamps=True)` can return word-level
# timestamps. If this fails to load, ASR still works -- we just fall back to
# plain (un-timestamped) transcription and skip the ts_*/subtitle outputs.
#
# Resolution order: explicit env var > a local `models/Qwen3-ForcedAligner-0.6B`
# folder if you've already downloaded it > the public Hugging Face Hub repo id
# (auto-downloaded on first use, given internet access -- e.g. on Colab).
# We deliberately do NOT default straight to the local folder path the way
# MODEL_ID historically did: if that folder doesn't exist, from_pretrained()
# fails immediately (it's not a valid HF repo id), and the app was silently
# falling back to timestamp-less transcription with no visible error.
_LOCAL_ALIGNER_DIR = BASE_DIR / "models" / "Qwen3-ForcedAligner-0.6B"
FORCED_ALIGNER_MODEL_ID = os.getenv("QWEN3_FORCED_ALIGNER_MODEL") or (
    str(_LOCAL_ALIGNER_DIR)
    if _LOCAL_ALIGNER_DIR.is_dir()
    else "Qwen/Qwen3-ForcedAligner-0.6B"
)

# Device selection: auto, cpu, cuda, cuda:0, mps.
DEVICE = os.getenv("QWEN3_ASR_DEVICE", "auto").lower()

# Torch dtype: auto, float16, bfloat16, float32.
TORCH_DTYPE = os.getenv("QWEN3_ASR_DTYPE", "auto").lower()

# Safe ASR audio sample rate. Qwen-family ASR models generally expect 16 kHz.
ASR_SAMPLE_RATE = int(os.getenv("QWEN3_ASR_SAMPLE_RATE", "16000"))

# Max number of new tokens the model may generate per transcription.
# Increase this for long-form audio, otherwise transcripts get truncated.
ASR_MAX_NEW_TOKENS = int(os.getenv("QWEN3_ASR_MAX_NEW_TOKENS", "2048"))

# Batch size cap used by qwen-asr's internal batching. -1 means unlimited.
ASR_MAX_BATCH_SIZE = int(os.getenv("QWEN3_ASR_MAX_BATCH_SIZE", "1"))

# Force a specific spoken language (e.g. "English", "Chinese"). Leave unset
# (None) to let Qwen3-ASR auto-detect the language.
ASR_LANGUAGE = os.getenv("QWEN3_ASR_LANGUAGE") or None

# Upload limits.
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "2048"))
MAX_CONTENT_LENGTH = MAX_UPLOAD_MB * 1024 * 1024

# Accepted upload extensions.
ALLOWED_EXTENSIONS = {
    "mp3",
    "wav",
    "m4a",
    "flac",
    "ogg",
    "mp4",
    "mkv",
    "mov",
    "webm",
    "avi",
}

# Controlled output file names inside each job folder.
ASR_AUDIO_FILENAME = "audio.wav"
TRANSCRIPT_FILENAME = "transcript.txt"          # unchanged: flat transcript text
MUTED_VIDEO_PREFIX = "video_no_audio"
SOURCE_PREFIX = "source"
TRANSLATED_FILENAME = "translated.txt"          # unchanged: flat translated text

# New: timestamped sibling files. Same content, plus per-segment start/end.
TS_TRANSCRIPT_FILENAME = "ts_transcript.json"
TS_TRANSLATED_FILENAME = "ts_translated.json"

# New: subtitle-burned video output.
SUBTITLED_VIDEO_PREFIX = "video_subtitled"

TRANSLATION_MODEL_ID = os.getenv(
    "TRANSLATION_MODEL_ID",
    "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
)
TRANSLATION_MAX_TOKENS = int(os.getenv("TRANSLATION_MAX_TOKENS", "2048"))
TRANSLATION_CHUNK_SIZE = int(os.getenv("TRANSLATION_CHUNK_SIZE", "4096"))

# --- Caption segmentation ------------------------------------------------
# Word-level timestamps from the forced aligner are grouped into
# subtitle-sized cues using these thresholds. A new cue starts whenever any
# one of them is exceeded, or whenever the previous word ended a sentence.
CAPTION_MAX_WORDS = int(os.getenv("CAPTION_MAX_WORDS", "8"))
CAPTION_MAX_CHARS = int(os.getenv("CAPTION_MAX_CHARS", "42"))
CAPTION_MAX_DURATION = float(os.getenv("CAPTION_MAX_DURATION", "4.0"))
CAPTION_MAX_GAP = float(os.getenv("CAPTION_MAX_GAP", "0.6"))

# --- Subtitle burn-in styling --------------------------------------------
# "Modern short-form captions" look: bold, white fill, blue accent outline.
# Drop a bold/heavy .ttf (e.g. Montserrat-ExtraBold, Poppins-Bold) into
# static/fonts/ and it will be picked up automatically via `fontsdir`; if
# nothing is there, libass falls back to whatever font by this name (or a
# close match) is installed on the system.
SUBTITLE_FONT_NAME = os.getenv("SUBTITLE_FONT_NAME", "Poppins ExtraBold")
SUBTITLE_FONT_DIR = str(BASE_DIR / "static" / "fonts")
SUBTITLE_FONT_SIZE_RATIO = float(os.getenv("SUBTITLE_FONT_SIZE_RATIO", "0.045"))
SUBTITLE_MARGIN_V_RATIO = float(os.getenv("SUBTITLE_MARGIN_V_RATIO", "0.08"))

# ASS colours use &HAABBGGRR (alpha, blue, green, red). Alpha 00 = opaque.
SUBTITLE_PRIMARY_COLOR = os.getenv("SUBTITLE_PRIMARY_COLOR", "&H00FFFFFF")  # white fill
SUBTITLE_OUTLINE_COLOR = os.getenv("SUBTITLE_OUTLINE_COLOR", "&H00FF802F")  # #2F80FF blue accent
SUBTITLE_BACK_COLOR = os.getenv("SUBTITLE_BACK_COLOR", "&H80000000")        # soft black shadow
SUBTITLE_OUTLINE_WIDTH = float(os.getenv("SUBTITLE_OUTLINE_WIDTH", "3.2"))
SUBTITLE_SHADOW = float(os.getenv("SUBTITLE_SHADOW", "0.8"))
