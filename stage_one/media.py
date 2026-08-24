import json
import shutil
import subprocess
from pathlib import Path

FFMPEG_PATH = shutil.which("ffmpeg")
FFPROBE_PATH = shutil.which("ffprobe")


class MediaError(RuntimeError):
    pass


def ffmpeg_available() -> bool:
    return bool(FFMPEG_PATH and FFPROBE_PATH)


def _require_ffmpeg() -> None:
    if not FFMPEG_PATH:
        raise MediaError(
            "FFmpeg was not found. Please install FFmpeg and restart the server."
        )

    if not FFPROBE_PATH:
        raise MediaError(
            "FFprobe was not found. Please install FFmpeg and restart the server."
        )


def _as_probe(probe_or_path):
    if isinstance(probe_or_path, dict):
        return probe_or_path
    return probe_media(probe_or_path)


def _run_ffmpeg(command, failure_message: str) -> None:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise MediaError(
            "FFmpeg was not found. Please install FFmpeg and restart the server."
        ) from exc
    except Exception as exc:
        raise MediaError(f"{failure_message}: {exc}") from exc

    if completed.returncode != 0:
        details = (completed.stderr or "").strip()
        if details:
            details = details.splitlines()[-1][:500]
            raise MediaError(f"{failure_message}: {details}")
        raise MediaError(failure_message)


def probe_media(input_path):
    _require_ffmpeg()

    input_path = Path(input_path)
    if not input_path.is_file():
        raise MediaError("Media file not found.")

    command = [
        FFPROBE_PATH,
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(input_path),
    ]

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise MediaError(
            "FFprobe was not found. Please install FFmpeg and restart the server."
        ) from exc
    except Exception as exc:
        raise MediaError(f"FFprobe failed: {exc}") from exc

    if completed.returncode != 0:
        details = (completed.stderr or "").strip()
        if details:
            details = details.splitlines()[-1][:500]
            raise MediaError(f"FFprobe failed: {details}")
        raise MediaError("FFprobe failed.")

    try:
        return json.loads(completed.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise MediaError("FFprobe returned invalid metadata.") from exc


def _streams(probe):
    if not isinstance(probe, dict):
        return []
    streams = probe.get("streams")
    if not isinstance(streams, list):
        return []
    return streams


def has_audio_stream(input_path) -> bool:
    probe = _as_probe(input_path)
    return any(stream.get("codec_type") == "audio" for stream in _streams(probe))


def is_video(input_path) -> bool:
    probe = _as_probe(input_path)
    return any(stream.get("codec_type") == "video" for stream in _streams(probe))


def extract_audio(input_path, output_path, sample_rate: int) -> None:
    _require_ffmpeg()

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.is_file():
        raise MediaError("Media file not found.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        FFMPEG_PATH,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        "-f",
        "wav",
        str(output_path),
    ]

    _run_ffmpeg(command, "Audio extraction failed")

    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise MediaError("Audio extraction did not produce a valid output file.")


def extract_muted_video(input_path, output_path) -> None:
    _require_ffmpeg()

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.is_file():
        raise MediaError("Media file not found.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    copy_command = [
        FFMPEG_PATH,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        "copy",
        str(output_path),
    ]

    try:
        _run_ffmpeg(copy_command, "Muted video stream copy failed")
        if output_path.is_file() and output_path.stat().st_size > 0:
            return
        raise MediaError("Muted video stream copy produced an empty file.")
    except MediaError:
        pass

    extension = output_path.suffix.lower()

    if extension == ".webm":
        video_codec = [
            "-c:v",
            "libvpx-vp9",
            "-crf",
            "32",
            "-b:v",
            "0",
        ]
    else:
        video_codec = [
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
        ]

    reencode_command = [
        FFMPEG_PATH,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-map",
        "0:v:0",
        "-an",
        *video_codec,
        str(output_path),
    ]

    _run_ffmpeg(reencode_command, "Muted video re-encode failed")

    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise MediaError("Muted video extraction did not produce a valid output file.")
