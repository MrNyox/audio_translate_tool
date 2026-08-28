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
import urllib.request
import urllib.error
from pathlib import Path
from typing import List, Tuple, Optional

import config
from stage_one import media

logger = logging.getLogger(__name__)
_FONT_FAMILY_CACHE: Optional[str] = None

# --- Online caption font ----------------------------------------------------
# Cairo is a single family purpose-built for bilingual Arabic/Latin UI text
# (it extends Titillium Web's Latin design with a matching Arabic Kufi-based
# design), so one font file covers both scripts cleanly with matching
# proportions/weight -- no font-fallback mismatch between an Arabic-only
# font and a separate Latin font. Google Fonts only ships it as a variable
# font, so we download the variable master once and pin/instantiate a
# single bold static weight (good contrast for burned-in captions, and
# avoids relying on the ffmpeg/libass build correctly picking a named
# instance out of a variable font, which many builds handle inconsistently).
_ONLINE_FONT_URL = (
    "https://raw.githubusercontent.com/google/fonts/main/ofl/cairo/"
    "Cairo%5Bslnt,wght%5D.ttf"
)
_ONLINE_FONT_VARIABLE_CACHE_NAME = "Cairo-Variable.ttf"
_ONLINE_FONT_STATIC_CACHE_NAME = "Cairo-ExtraBold-Static.ttf"
_ONLINE_FONT_WEIGHT = 800  # ExtraBold: bold/legible enough for on-video captions
_ONLINE_FONT_DOWNLOAD_TIMEOUT = 12  # seconds


def _instantiate_static_weight(variable_path: Path, static_path: Path, weight: int) -> bool:
    """Pin a variable font to a single static weight instance via fontTools.

    Returns True on success. Any failure (fontTools not installed, corrupt
    font, etc.) is logged and treated as non-fatal -- callers fall back to
    the bundled/local font instead.
    """
    try:
        from fontTools.ttLib import TTFont
        from fontTools.varLib.instancer import instantiateVariableFont
    except ImportError:
        logger.warning(
            "fonttools is not installed, so the downloaded variable Cairo "
            "font can't be pinned to a static weight. Add 'fonttools' to "
            "requirements.txt to enable the online caption font."
        )
        return False

    try:
        font = TTFont(str(variable_path))
        axes = {"wght": weight}
        if "slnt" in {a.axisTag for a in font["fvar"].axes} if "fvar" in font else False:
            axes["slnt"] = 0
        static_font = instantiateVariableFont(font, axes, inplace=False)
        tmp_path = static_path.with_suffix(".tmp")
        static_font.save(str(tmp_path))
        tmp_path.replace(static_path)
        return True
    except Exception:
        logger.exception("Failed to instantiate static weight from %s", variable_path)
        return False


def _download_online_caption_font(font_dir: Path) -> Optional[Path]:
    """
    Ensure a bilingual Arabic/Latin caption font is available in
    `font_dir`, downloading it from Google Fonts on first use and caching
    the result on disk so subsequent runs don't re-fetch it.

    Returns the path to a ready-to-use static .ttf, or None if the font
    could not be obtained (no internet, blocked domain, fontTools missing,
    etc.) -- in which case the caller should fall back to a bundled/local
    font so subtitle rendering still works, just without this specific look.
    """
    font_dir.mkdir(parents=True, exist_ok=True)
    static_path = font_dir / _ONLINE_FONT_STATIC_CACHE_NAME

    if static_path.is_file() and static_path.stat().st_size > 0:
        return static_path

    variable_path = font_dir / _ONLINE_FONT_VARIABLE_CACHE_NAME

    try:
        logger.info("Downloading online caption font from %s", _ONLINE_FONT_URL)
        request = urllib.request.Request(
            _ONLINE_FONT_URL, headers={"User-Agent": "audio_translate_tool/1.0"}
        )
        with urllib.request.urlopen(request, timeout=_ONLINE_FONT_DOWNLOAD_TIMEOUT) as resp:
            data = resp.read()

        if not data:
            raise ValueError("Downloaded font file was empty.")

        tmp_variable_path = variable_path.with_suffix(".tmp")
        tmp_variable_path.write_bytes(data)
        tmp_variable_path.replace(variable_path)

    except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError) as exc:
        logger.warning(
            "Could not download online caption font (%s). Falling back to "
            "the bundled/local subtitle font instead.",
            exc,
        )
        return None

    if not _instantiate_static_weight(variable_path, static_path, _ONLINE_FONT_WEIGHT):
        return None

    logger.info("Online caption font ready: %s", static_path)
    return static_path


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

    # Preferred: a clean bilingual Arabic/Latin font fetched online (Cairo).
    # This is best-effort -- any failure (offline, blocked domain, missing
    # fonttools) falls straight through to the bundled/local font below, so
    # subtitle rendering keeps working even with no internet access.
    if config.SUBTITLE_USE_ONLINE_FONT:
        online_font_path = _download_online_caption_font(font_dir)
        if online_font_path is not None:
            detected_family = _read_font_family(online_font_path)
            if detected_family:
                logger.info(
                    "Using online caption font family '%s' from: %s",
                    detected_family,
                    online_font_path,
                )
                _FONT_FAMILY_CACHE = detected_family
                return _FONT_FAMILY_CACHE

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


def segments_to_ass(
    segments: List[dict],
    video_width: int,
    video_height: int,
    *,
    apply_pacing: bool = True,
    max_words: int = 7,
    target_wps: float = 2.5,
    min_duration: float = 0.8,
    min_gap: float = 0.08,
) -> str:
    """Build a complete .ass subtitle file from translated segments.

    By default this re-times the segments through `normalize_subtitle_pacing`
    first. Raw segment start/end windows come from the *original* speech's
    timing and say nothing about how long the (possibly translated) caption
    text actually takes to read -- a short window with a lot of text would
    otherwise flash by too fast, and a long window with little text would
    otherwise linger on screen well after the speaker has moved on. Pass
    `apply_pacing=False` only if `segments` has already been normalized
    upstream and you want to avoid re-chunking it a second time.
    """

    if apply_pacing:
        segments = normalize_subtitle_pacing(
            segments,
            max_words=max_words,
            target_wps=target_wps,
            min_duration=min_duration,
            min_gap=min_gap,
        )

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

# Punctuation that makes a natural place to break a line (English + Arabic).
_STRONG_BREAK_CHARS = ".!?؟!،؛:,"


def _split_into_word_chunks(text: str, max_words: int) -> list:
    """Split text into chunks of at most `max_words` words, preferring to
    end a chunk on a punctuation mark near the word-count limit so breaks
    land on natural clause/sentence boundaries instead of mid-phrase.
    """
    words = text.split()
    if len(words) <= max_words:
        return [text]

    chunks = []
    i = 0
    n = len(words)

    while i < n:
        window_end = min(i + max_words, n)

        if window_end >= n:
            chosen_end = n
        else:
            # Scan backward from the window edge for a word ending in
            # punctuation, so we don't cut a sentence off arbitrarily.
            chosen_end = window_end
            for j in range(window_end, i + 1, -1):
                if words[j - 1] and words[j - 1][-1] in _STRONG_BREAK_CHARS:
                    chosen_end = j
                    break

        # Avoid leaving an orphan single word behind for the next chunk
        # unless it's genuinely the end of the text.
        if chosen_end - i < 2 and window_end < n:
            chosen_end = window_end

        chunks.append(" ".join(words[i:chosen_end]))
        i = chosen_end

    return chunks


def normalize_subtitle_pacing(
    segments: List[dict],
    max_words: int = 7,
    target_wps: float = 2.5,
    min_duration: float = 0.8,
    min_gap: float = 0.08,
) -> List[dict]:
    """
    Re-times subtitle segments so they're comfortable to read at normal
    human reading speed, and never race ahead of the speech they belong to.

    Two things happen here, and both matter:

    1. Long lines are capped at `max_words` per cue, breaking on natural
       punctuation where possible (same as before).
    2. Each cue is given a duration based on `target_wps` (~150 words/min,
       a normal comfortable reading pace) rather than just dividing the
       original segment's time window evenly. Simply chopping a segment
       into more pieces without changing its total duration doesn't
       actually slow down the reading pace -- word count and time shrink
       together, so words-per-second stays identical. If a translation
       is longer than what the original segment's timestamps allow, this
       borrows the natural pause before the *next* segment's speech
       starts (never past it) to stretch the reading time -- the same
       approach editing tools like Premiere use for auto-captions. Only
       when there's no gap left to borrow does it fall back to
       compressing the pace, which is logged so it's visible when a
       translation is simply too verbose for the available speech time.
    """
    normalized = []
    n = len(segments)

    for idx, seg in enumerate(segments):
        start = float(seg.get("start", 0.0) or 0.0)
        end_raw = seg.get("end")
        end = float(end_raw) if end_raw is not None else start
        text = str(seg.get("text", "")).strip()

        if not text:
            continue

        if end <= start:
            end = start + min_duration

        # The next segment's original start time is the hard ceiling this
        # cue's chunks may stretch into -- we never want translated text
        # to still be on screen once the next line is actually being said.
        hard_ceiling = None
        if idx + 1 < n:
            try:
                next_start_raw = segments[idx + 1].get("start")
                next_start = float(next_start_raw) if next_start_raw is not None else None
            except Exception:
                next_start = None
            if next_start is not None and next_start > start:
                hard_ceiling = next_start - min_gap
        else:
            # Last segment: there's no next cue to avoid colliding with, and
            # this function doesn't know the video's total duration. Cap the
            # borrowed extension to double the original slot so a very
            # verbose final translation still compresses instead of
            # potentially running past the end of the video.
            hard_ceiling = start + max(end - start, min_duration) * 2

        chunks = _split_into_word_chunks(text, max_words)
        chunk_word_counts = [max(1, len(c.split())) for c in chunks]
        ideal_durations = [max(min_duration, wc / target_wps) for wc in chunk_word_counts]
        total_ideal = sum(ideal_durations)

        # Available window: start with the original segment window, then
        # borrow from the gap before the next line only as much as needed
        # (and never more than what's actually free).
        available_end = end
        if hard_ceiling is not None:
            if hard_ceiling > available_end:
                available_end = min(hard_ceiling, start + total_ideal)
            else:
                available_end = min(available_end, hard_ceiling)

        available_time = max(available_end - start, min_duration)

        if total_ideal > available_time:
            logger.info(
                "Subtitle segment at %.2fs is too verbose for its available "
                "speech window (%d words need ~%.1fs, only %.1fs available); "
                "compressing reading pace to stay in sync.",
                start, sum(chunk_word_counts), total_ideal, available_time,
            )
            scale = available_time / total_ideal
        else:
            scale = 1.0

        cursor = start
        for chunk_text, ideal in zip(chunks, ideal_durations):
            duration = max(ideal * scale, 0.35)  # absolute floor: always visible long enough to register
            chunk_start = cursor
            chunk_end = chunk_start + duration

            if hard_ceiling is not None:
                chunk_end = min(chunk_end, hard_ceiling)
                if chunk_end <= chunk_start:
                    chunk_end = chunk_start + 0.35

            normalized.append({
                "start": round(chunk_start, 3),
                "end": round(chunk_end, 3),
                "text": chunk_text.strip(),
            })
            cursor = chunk_end

    return normalized


def burn_subtitles(video_path, ass_path, output_path, audio_source_path=None) -> None:
    """Burn a .ass subtitle file into a video, re-encoding video only.

    `audio_source_path`, if given and it actually contains an audio stream,
    is muxed back in as the output's audio track -- `video_path` here is
    the *muted* video from Stage One (audio was stripped out separately for
    ASR), so without this the final "subtitled" video would be silent even
    though the original clip had audio.
    """
    if not media.ffmpeg_available():
        raise media.MediaError(
            "FFmpeg was not found. Please install FFmpeg and restart the server."
        )
    media.check_arabic_rendering_support()

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
        audio_codec = ["-c:a", "libopus", "-b:a", "160k"]
    else:
        video_codec = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]
        audio_codec = ["-c:a", "aac", "-b:a", "192k"]

    audio_source_path = Path(audio_source_path) if audio_source_path else None
    include_audio = bool(
        audio_source_path
        and audio_source_path.is_file()
        and media.has_audio_stream(audio_source_path)
    )

    command = [
        media.FFMPEG_PATH,
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-i", str(video_path),
    ]

    if include_audio:
        command += ["-i", str(audio_source_path)]

    command += ["-map", "0:v:0"]  # Explicitly select ONLY the first video stream

    if include_audio:
        command += ["-map", "1:a:0", "-shortest"]

    command += ["-vf", filter_arg, *video_codec]

    if include_audio:
        command += audio_codec
    else:
        command += ["-an"]

    command += ["-sn", str(output_path)]  # Remove all subtitle streams (force only burned-in subs)

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
