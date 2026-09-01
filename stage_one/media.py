import json
import logging
import shutil
import subprocess
from pathlib import Path

FFMPEG_PATH = shutil.which("ffmpeg")
FFPROBE_PATH = shutil.which("ffprobe")


class MediaError(RuntimeError):
    pass


def ffmpeg_available() -> bool:
    return bool(FFMPEG_PATH and FFPROBE_PATH)


_ARABIC_RENDER_LIBS = ("libass", "libfribidi", "libharfbuzz", "libfontconfig", "libfreetype")
_arabic_support_checked = False


def check_arabic_rendering_support() -> None:
    """
    Log a loud, explicit warning if the installed ffmpeg's `ass` filter was
    NOT built with the libraries required to correctly shape/join Arabic
    (and other RTL/complex) script glyphs:

      - libass        : renders .ass subtitles at all
      - libfribidi    : bidi algorithm (right-to-left ordering)
      - libharfbuzz   : complex-script glyph shaping (letter joining)
      - libfontconfig : lets libass resolve/load fonts by family name
      - libfreetype   : rasterizes the glyph outlines

    Without fribidi/harfbuzz specifically, Arabic text will often render as
    disconnected, wrongly-ordered, or completely missing glyphs instead of
    properly joined right-to-left script -- which looks identical to "the
    subtitles just aren't there." This only *detects and reports* the
    problem (once, cached); it can't fix a broken ffmpeg build, but a clear
    log line is far better than mysteriously blank captions.
    """
    global _arabic_support_checked
    if _arabic_support_checked or not FFMPEG_PATH:
        return
    _arabic_support_checked = True

    try:
        completed = subprocess.run(
            [FFMPEG_PATH, "-hide_banner", "-version"],
            capture_output=True,
            text=True,
            check=False,
        )
        config_output = (completed.stdout or "") + (completed.stderr or "")
    except Exception:
        logging.getLogger(__name__).warning(
            "Could not run 'ffmpeg -version' to verify Arabic/RTL subtitle "
            "rendering support."
        )
        return

    missing = [
        lib for lib in _ARABIC_RENDER_LIBS
        if f"--enable-{lib}" not in config_output
    ]

    if missing:
        logging.getLogger(__name__).warning(
            "This ffmpeg build is missing: %s. Arabic (and other RTL/complex "
            "script) subtitles may render blank, disconnected, or in the "
            "wrong order even though the .ass file itself is correct. "
            "Install a full ffmpeg build (on Debian/Ubuntu: "
            "'apt-get install ffmpeg', which normally already includes "
            "these; on conda-forge use the 'ffmpeg' feedstock, not a "
            "minimal static binary) to fix this.",
            ", ".join(missing),
        )
    else:
        logging.getLogger(__name__).info(
            "ffmpeg build has full Arabic/RTL subtitle rendering support "
            "(%s all present).", ", ".join(_ARABIC_RENDER_LIBS)
        )


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
