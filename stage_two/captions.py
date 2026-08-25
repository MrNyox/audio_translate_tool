"""Subtitle rendering.

Turns timestamped, translated segments (ts_translated.json) into a styled
.ass subtitle file, then burns it into the muted video from Stage One using
ffmpeg's `ass` filter (libass). Style target: bold, high-contrast "modern
short-form" captions -- white fill with a blue accent outline, similar to
the on-screen captions common on TikTok/Reels/Shorts.

TTS dubbing is intentionally out of scope here -- this module only ever
produces a video with hard-coded (burned-in) subtitle text over the
original visual track.
"""

import subprocess
from pathlib import Path
from typing import List, Tuple

import config
from stage_one import media


def video_dimensions(probe: dict) -> Tuple[int, int]:
    """Best-effort (width, height) from an ffprobe result. Falls back to a
    common 720p frame if nothing usable is found."""
    streams = probe.get("streams") if isinstance(probe, dict) else None
    for stream in streams or []:
        if stream.get("codec_type") == "video":
            width = stream.get("width")
            height = stream.get("height")
            if width and height:
                return int(width), int(height)
    return 1280, 720


def _format_ass_time(seconds: float) -> str:
    # Work in whole centiseconds throughout so a value like 59.999s rounds
    # up into the next minute/hour correctly instead of producing "60.00"
    # seconds (which several players/libass will refuse to parse).
    total_centis = max(0, int(round(float(seconds) * 100)))

    hours, remainder = divmod(total_centis, 360000)
    minutes, remainder = divmod(remainder, 6000)
    secs, centis = divmod(remainder, 100)

    return f"{hours:d}:{minutes:02d}:{secs:02d}.{centis:02d}"


def _escape_ass_text(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("\n", "\\N")
    )


def segments_to_ass(segments: List[dict], video_width: int, video_height: int) -> str:
    """Build a complete .ass subtitle file from translated segments."""
    font_size = max(18, int(round(video_height * config.SUBTITLE_FONT_SIZE_RATIO)))
    margin_v = max(20, int(round(video_height * config.SUBTITLE_MARGIN_V_RATIO)))

    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {video_width}\n"
        f"PlayResY: {video_height}\n"
        "WrapStyle: 2\n"
        "ScaledBorderAndShadow: yes\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        "Style: Caption,"
        f"{config.SUBTITLE_FONT_NAME},{font_size},"
        f"{config.SUBTITLE_PRIMARY_COLOR},{config.SUBTITLE_PRIMARY_COLOR},"
        f"{config.SUBTITLE_OUTLINE_COLOR},{config.SUBTITLE_BACK_COLOR},"
        f"-1,0,0,0,100,100,0,0,1,"
        f"{config.SUBTITLE_OUTLINE_WIDTH},{config.SUBTITLE_SHADOW},2,"
        f"60,60,{margin_v},1\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    lines = [header]
    for seg in segments:
        start_value = float(seg.get("start") or 0.0)
        end_value = seg.get("end")
        end_value = float(end_value) if end_value is not None else start_value
        if end_value <= start_value:
            end_value = start_value + 0.5  # guarantee a visible, non-zero cue

        text = _escape_ass_text(str(seg.get("text", "")).strip())
        if not text:
            continue

        start = _format_ass_time(start_value)
        end = _format_ass_time(end_value)
        lines.append(f"Dialogue: 0,{start},{end},Caption,,0,0,0,,{text}\n")

    return "".join(lines)


def _escape_ffmpeg_filter_path(path: Path) -> str:
    """Escape a path for safe use inside an ffmpeg filtergraph argument."""
    escaped = str(path)
    escaped = escaped.replace("\\", "\\\\")
    escaped = escaped.replace(":", "\\:")
    escaped = escaped.replace("'", "\\'")
    return escaped


def burn_subtitles(video_path, ass_path, output_path) -> None:
    """Burn a .ass subtitle file into a video, re-encoding video only."""
    if not media.ffmpeg_available():
        raise media.MediaError(
            "FFmpeg was not found. Please install FFmpeg and restart the server."
        )

    video_path = Path(video_path)
    ass_path = Path(ass_path)
    output_path = Path(output_path)

    if not video_path.is_file():
        raise media.MediaError("Muted video file not found for subtitle burn-in.")
    if not ass_path.is_file():
        raise media.MediaError("Subtitle (.ass) file not found for subtitle burn-in.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    filter_arg = f"ass={_escape_ffmpeg_filter_path(ass_path)}"
    font_dir = Path(config.SUBTITLE_FONT_DIR)
    if font_dir.is_dir():
        filter_arg += f":fontsdir={_escape_ffmpeg_filter_path(font_dir)}"

    extension = output_path.suffix.lower()
    if extension == ".webm":
        video_codec = ["-c:v", "libvpx-vp9", "-crf", "32", "-b:v", "0"]
    else:
        video_codec = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]

    command = [
        media.FFMPEG_PATH,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        filter_arg,
        *video_codec,
        "-an",
        str(output_path),
    ]

    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise media.MediaError(
            "FFmpeg was not found. Please install FFmpeg and restart the server."
        ) from exc
    except Exception as exc:
        raise media.MediaError(f"Subtitle burn-in failed: {exc}") from exc

    if completed.returncode != 0:
        details = (completed.stderr or "").strip()
        if details:
            details = details.splitlines()[-1][:500]
            raise media.MediaError(f"Subtitle burn-in failed: {details}")
        raise media.MediaError("Subtitle burn-in failed.")

    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise media.MediaError("Subtitle burn-in did not produce a valid output file.")
