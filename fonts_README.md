# Subtitle fonts

Drop a bold/heavy `.ttf` or `.otf` here to control how the burned-in
subtitles look (see `stage_two/captions.py` + `config.SUBTITLE_FONT_NAME`).

For a "modern short-form captions" (TikTok/Reels/Shorts) look, a heavy
weight works best, e.g.:

- Montserrat ExtraBold / Black
- Poppins ExtraBold / Black
- Archivo Black

Set `config.SUBTITLE_FONT_NAME` (or the `SUBTITLE_FONT_NAME` env var) to
match the font's internal family name exactly (not just the filename) --
e.g. `Poppins ExtraBold`, not `Poppins-ExtraBold.ttf`.

If this folder is empty, ffmpeg/libass will fall back to searching the
system's installed fonts for the closest match to `SUBTITLE_FONT_NAME`,
which may not look the same across machines.
