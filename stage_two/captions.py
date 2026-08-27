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

import logging
import struct
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional

import config
from stage_one import media

logger = logging.getLogger(__name__)
_FONT_FAMILY_CACHE: Optional[str] = None


def _read_font_family(path: Path) -> Optional[str]:
    """
    Read the internal font family name from a .ttf/.otf/.ttc file.

    This is important because the file name does not matter to libass.
    The ASS `Fontname` field must match the internal font family name.
    """
    try:
        data = path.read_bytes()
        if not data:
            return None

        # TrueType Collection support: use the first font in the collection.
        if data[:4] == b"ttcf":
            if len(data) < 16:
                return None
            offset = struct.unpack(">I", data[12:16])[0]
        else:
            offset = 0

        if len(data) < offset + 12:
            return None

        num_tables = struct.unpack(">H", data[offset + 4:offset + 6])[0]
        pos = offset + 12
        name_offset = None

        # Find the 'name' table.
        for _ in range(num_tables):
            if len(data) < pos + 16:
                break

            tag = data[pos:pos + 4]
            if tag == b"name":
                name_offset = struct.unpack(">I", data[pos + 8:pos + 12])[0]
                break

            pos += 16

        if name_offset is None or len(data) < name_offset + 6:
            return None

        count = struct.unpack(">H", data[name_offset + 2:name_offset + 4])[0]
        string_offset = struct.unpack(">H", data[name_offset + 4:name_offset + 6])[0]

        strings_start = name_offset + string_offset
        rec = name_offset + 6
        candidates = []

        for _ in range(count):
            if len(data) < rec + 12:
                break

            platform_id, encoding_id, language_id, name_id, length, offset = struct.unpack(
                ">HHHHHH",
                data[rec:rec + 12]
            )

            # name_id 1  = Font Family
            # name_id 16 = Typographic Family
            if name_id in (1, 16):
                start = strings_start + offset
                raw = data[start:start + length]

                try:
                    if platform_id in (0, 3):
                        # Unicode / Windows platforms are UTF-16BE.
                        s = raw.decode("utf-16-be", "ignore")
                    elif platform_id == 1:
                        # Mac platform. Try UTF-8 first, then Latin-1.
                        s = raw.decode("utf-8", "ignore")
                        if not s:
                            s = raw.decode("latin-1", "ignore")
                    else:
                        s = raw.decode("utf-8", "ignore")
                except Exception:
                    s = ""

                s = s.strip()
                if s:
                    priority = 0

                    # Prefer English Windows names if available.
                    if platform_id == 3 and language_id == 0x0409:
                        priority = 3
                    elif platform_id == 0:
                        priority = 2
                    elif platform_id == 1 and language_id == 0:
                        priority = 1

                    # Prefer typographic family name if available.
                    candidates.append((priority, name_id == 16, s))

            rec += 12

        if not candidates:
            return None

        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return candidates[0][2]

    except Exception:
        logger.exception("Failed to read font family from %s", path)
        return None


def _resolve_subtitle_font_family() -> str:
    """
    Resolve the font family name that should be written into the ASS file.

    If a local font file exists in config.SUBTITLE_FONT_DIR, read its real
    internal family name. This prevents failures caused by the configured
    font name not exactly matching the font file's internal name.
    """
    global _FONT_FAMILY_CACHE

    if _FONT_FAMILY_CACHE is not None:
        return _FONT_FAMILY_CACHE

    family = config.SUBTITLE_FONT_NAME
    font_dir = Path(config.SUBTITLE_FONT_DIR)

    if font_dir.is_dir():
        valid_suffixes = {".ttf", ".otf", ".ttc"}

        font_files = [
            p
            for p in sorted(font_dir.iterdir())
            if p.is_file() and p.suffix.lower() in valid_suffixes
        ]

        # If multiple font files exist, prefer Arabic/Plex-looking files first.
        font_files.sort(
            key=lambda p: (
                0 if "arabic" in p.name.lower() else 1,
                0 if "plex" in p.name.lower() else 1,
                p.name,
            )
        )

        for font_file in font_files:
            detected_family = _read_font_family(font_file)
            if detected_family:
                logger.info(
                    "Detected subtitle font family '%s' from file: %s",
                    detected_family,
                    font_file,
                )
                family = detected_family
                break

    _FONT_FAMILY_CACHE = family
    return family


def _segment_value(seg: dict, *keys, default=None):
    """
    Get a value from a segment dictionary while tolerating accidental
    trailing/leading spaces in keys, e.g. "start " instead of "start".
    """
    if not isinstance(seg, dict):
        return default

    normalized = {}
    for key, value in seg.items():
        if isinstance(key, str):
            normalized[key.strip()] = value

    for key in keys:
        if key in seg and seg[key] is not None:
            return seg[key]

        if key in normalized and normalized[key] is not None:
            return normalized[key]

    return default

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

    font_name = _resolve_subtitle_font_family()

    logger.info(
        "Generating ASS subtitles using font family: '%s'",
        font_name,
    )

    font_size = max(18, int(round(video_height * config.SUBTITLE_FONT_SIZE_RATIO)))
    margin_v = max(20, int(round(video_height * config.SUBTITLE_MARGIN_V_RATIO)))

    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {video_width}\n"
        f"PlayResY: {video_height}\n"
        "WrapStyle: 0\n"
        "ScaledBorderAndShadow: yes\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        "Style: Caption,"
        f"{font_name},{font_size},"
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
        try:
            start_value = float(
                _segment_value(seg, "start", "start_time", default=0.0) or 0.0
            )
        except Exception:
            start_value = 0.0

        end_raw = _segment_value(seg, "end", "end_time", default=None)

        try:
            end_value = float(end_raw) if end_raw is not None else start_value
        except Exception:
            end_value = start_value

        if end_value <= start_value:
            end_value = start_value + 0.5  # guarantee a visible, non-zero cue

        raw_text = _segment_value(seg, "text", default="")
        text = _escape_ass_text(str(raw_text or "").strip())

        if not text:
            continue

        start = _format_ass_time(start_value)
        end = _format_ass_time(end_value)

        lines.append(
            f"Dialogue: 0,{start},{end},Caption,,0,0,0,,{text}\n"
        )

    # UTF-8 BOM helps some subtitle parsers/renderers detect UTF-8 correctly.
    return "\ufeff" + "".join(lines)


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
        "-loglevel", "error",
        "-y",
        "-i", str(video_path),
        "-map", "0:v:0",  # Explicitly select ONLY the first video stream
        "-vf", filter_arg,
        *video_codec,
        "-sn",            # Remove all subtitle streams (force only burned-in subs)
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
