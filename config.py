import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Output root. Automatically created at runtime by app.py.
JOB_OUTPUT_ROOT = Path(
    os.getenv("STAGE_ONE_OUTPUT_ROOT", str(BASE_DIR / "job_output"))
).resolve()

# Qwen3-ASR model id or local folder path.
MODEL_ID = os.getenv(
    "QWEN3_ASR_MODEL",
    str(BASE_DIR / "models" / "Qwen3-ASR-1.7B")
)

# Device selection: auto, cpu, cuda, cuda:0, mps.
DEVICE = os.getenv("QWEN3_ASR_DEVICE", "auto").lower()

# Torch dtype: auto, float16, bfloat16, float32.
TORCH_DTYPE = os.getenv("QWEN3_ASR_DTYPE", "auto").lower()

# Safe ASR audio sample rate. Qwen-family ASR models generally expect 16 kHz.
ASR_SAMPLE_RATE = int(os.getenv("QWEN3_ASR_SAMPLE_RATE", "16000"))

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
TRANSCRIPT_FILENAME = "transcript.txt"
MUTED_VIDEO_PREFIX = "video_no_audio"
SOURCE_PREFIX = "source"
