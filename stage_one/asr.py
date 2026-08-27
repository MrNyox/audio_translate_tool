import logging
import threading
from pathlib import Path
import gc
import config
import torch

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_inference_lock = threading.Lock()

_model = None
_device = None
_aligner_loaded = False


def is_model_loaded() -> bool:
    return _model is not None


def is_aligner_loaded() -> bool:
    return _aligner_loaded


def unload() -> None:
    """Unload the ASR model and free GPU memory."""
    global _model, _device, _aligner_loaded
    with _lock:
        _model = None
        _device = None
        _aligner_loaded = False
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _resolve_device(torch_module):
    requested = (config.DEVICE or "auto").lower()

    if requested == "auto":
        if torch_module.cuda.is_available():
            return "cuda:0"

        mps_available = (
            hasattr(torch_module, "backends")
            and hasattr(torch_module.backends, "mps")
            and torch_module.backends.mps.is_available()
        )
        if mps_available:
            return "mps"

        return "cpu"

    if requested.startswith("cuda") and not torch_module.cuda.is_available():
        return "cpu"

    if requested == "cuda":
        return "cuda:0"

    if requested == "mps":
        mps_available = (
            hasattr(torch_module, "backends")
            and hasattr(torch_module.backends, "mps")
            and torch_module.backends.mps.is_available()
        )
        if not mps_available:
            return "cpu"

    return requested


def _resolve_torch_dtype(torch_module, device: str):
    requested = (config.TORCH_DTYPE or "auto").lower()

    if requested in {"", "auto"}:
        if device.startswith("cuda"):
            # bfloat16 requires Ampere (compute capability 8.0) or newer.
            # T4 and older GPUs (compute capability 7.x) will crash or
            # fail to load certain models (like Qwen3-ForcedAligner) if
            # forced to use bfloat16 in custom CUDA kernels.
            if hasattr(torch_module.cuda, "is_bf16_supported") and torch_module.cuda.is_bf16_supported():
                return torch_module.bfloat16
            return torch_module.float16
        return torch_module.float32

    mapping = {
        "float16": torch_module.float16,
        "fp16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "bf16": torch_module.bfloat16,
        "float32": torch_module.float32,
        "fp32": torch_module.float32,
    }

    dtype = mapping.get(requested, torch_module.float32)

    # Avoid float16/bfloat16 on CPU unless explicitly forced by the user.
    if device == "cpu" and requested in {"", "auto"} and dtype != torch_module.float32:
        return torch_module.float32

    return dtype


def load_model() -> None:
    """Load the Qwen3-ASR model using the official `qwen-asr` package.

    Qwen3-ASR-1.7B uses a custom `qwen3_asr` architecture. It is NOT a
    standard seq2seq speech model, so it can't be loaded through
    `AutoModelForSpeechSeq2Seq` + `transformers.pipeline("automatic-speech-recognition", ...)`.
    It must be loaded through `qwen_asr.Qwen3ASRModel`, which wraps the
    transformers (or vLLM) backend correctly.
    See: https://huggingface.co/Qwen/Qwen3-ASR-1.7B

    We also try to load `Qwen3-ForcedAligner-0.6B` alongside it so that
    `.transcribe(..., return_time_stamps=True)` can return word-level
    timestamps. If the aligner can't be loaded (missing weights, missing
    package support, OOM, etc.) we retry without it -- plain transcription
    must keep working even if timestamps aren't available.
    """
    global _model, _device, _aligner_loaded

    with _lock:
        if _model is not None:
            return

        try:
            import torch
            from qwen_asr import Qwen3ASRModel
        except Exception as exc:
            raise RuntimeError(
                "Qwen3-ASR dependencies are missing. Install them with "
                "`pip install -U qwen-asr torch soundfile`."
            ) from exc

        device = _resolve_device(torch)
        torch_dtype = _resolve_torch_dtype(torch, device)

        try:
            model = Qwen3ASRModel.from_pretrained(
                config.MODEL_ID,
                dtype=torch_dtype,
                device_map=device,
                max_inference_batch_size=config.ASR_MAX_BATCH_SIZE,
                max_new_tokens=config.ASR_MAX_NEW_TOKENS,
                forced_aligner=config.FORCED_ALIGNER_MODEL_ID,
                forced_aligner_kwargs=dict(
                    dtype=torch_dtype,
                    device_map=device,
                ),
            )

            _model = model
            _device = device
            _aligner_loaded = True
            return

        except Exception as aligner_exc:
            # Fall back to ASR-only so transcription keeps working even if
            # the forced aligner model isn't available locally / fails to
            # load. We just won't get ts_transcript.json / ts_translated.json
            # / subtitles for this run. Log this LOUDLY (not just a debug
            # line) -- silently swallowing it made a real load failure look
            # identical to "aligner intentionally not configured", which is
            # exactly what caused ts_transcript.json to ship empty with no
            # visible explanation.
            logger.error(
                "Forced aligner failed to load from '%s' -- continuing "
                "WITHOUT word-level timestamps (ts_transcript.json will be "
                "empty and subtitle burn-in will be skipped for this run). "
                "Set QWEN3_FORCED_ALIGNER_MODEL to a valid local path or HF "
                "Hub repo id, or download the model into "
                "models/Qwen3-ForcedAligner-0.6B/, to fix this. "
                "Underlying error: %s",
                config.FORCED_ALIGNER_MODEL_ID,
                aligner_exc,
                exc_info=True,
            )
            try:
                model = Qwen3ASRModel.from_pretrained(
                    config.MODEL_ID,
                    dtype=torch_dtype,
                    device_map=device,
                    max_inference_batch_size=config.ASR_MAX_BATCH_SIZE,
                    max_new_tokens=config.ASR_MAX_NEW_TOKENS,
                )

                _model = model
                _device = device
                _aligner_loaded = False
                return

            except Exception as exc:
                _model = None
                _device = None
                _aligner_loaded = False
                raise RuntimeError(f"Failed to load Qwen3-ASR model: {exc}") from exc


def _word_field(word, *names, default=None):
    """Read an attribute/key from a timestamp entry, tolerating either an
    object with attributes (dataclass-like) or a plain dict, and slightly
    different field naming across qwen-asr versions."""
    for name in names:
        if hasattr(word, name):
            value = getattr(word, name)
            if value is not None:
                return value
        if isinstance(word, dict) and word.get(name) is not None:
            return word[name]
    return default


def _group_word_stamps(word_stamps):
    """Group word-level timestamps into subtitle-sized cues.

    A new cue is started whenever any of the configured thresholds
    (word count, character count, cue duration, silence gap) is exceeded,
    or whenever the previous word ends a sentence (., ?, !, ...).
    """
    max_words = config.CAPTION_MAX_WORDS
    max_chars = config.CAPTION_MAX_CHARS
    max_duration = config.CAPTION_MAX_DURATION
    max_gap = config.CAPTION_MAX_GAP

    segments = []
    current = []

    def flush():
        if not current:
            return
        text = " ".join(w["text"] for w in current).strip()
        if not text:
            current.clear()
            return
        segments.append(
            {
                "start": round(current[0]["start"], 3),
                "end": round(current[-1]["end"], 3),
                "text": text,
            }
        )
        current.clear()

    for raw_word in word_stamps:
        text = str(_word_field(raw_word, "text", "word", default="")).strip()
        if not text:
            continue

        start = float(_word_field(raw_word, "start_time", "start", default=0.0))
        end = float(_word_field(raw_word, "end_time", "end", default=start))
        if end < start:
            end = start

        if current:
            gap = start - current[-1]["end"]
            prev_ends_sentence = current[-1]["text"].endswith((".", "?", "!", "\u2026", "\u061f"))
            proposed_words = len(current) + 1
            proposed_chars = len(" ".join(w["text"] for w in current)) + 1 + len(text)
            proposed_duration = end - current[0]["start"]

            if (
                prev_ends_sentence
                or gap > max_gap
                or proposed_words > max_words
                or proposed_chars > max_chars
                or proposed_duration > max_duration
            ):
                flush()

        current.append({"text": text, "start": start, "end": end})

    flush()
    return segments


def _run_transcription(audio_path, with_timestamps: bool) -> dict:
    audio_path = Path(audio_path)

    if not audio_path.is_file():
        raise RuntimeError("Extracted audio file not found.")

    load_model()
    want_timestamps = with_timestamps and _aligner_loaded

    with _inference_lock:
        try:
            results = _model.transcribe(
                audio=str(audio_path),
                language=config.ASR_LANGUAGE,
                return_time_stamps=want_timestamps,
            )
        except Exception as exc:
            raise RuntimeError(f"Transcription failed: {exc}") from exc

    if not results:
        return {"text": "", "segments": []}

    result = results[0]
    text = getattr(result, "text", None)
    if text is None and isinstance(result, dict):
        text = result.get("text")
    text = str(text or "").strip()

    segments = []
    if want_timestamps:
        raw_stamps = getattr(result, "time_stamps", None)
        if raw_stamps is None and isinstance(result, dict):
            raw_stamps = result.get("time_stamps")

        word_stamps = []
        if raw_stamps:
            # `time_stamps` is batched (one list per input audio); we only
            # ever pass a single audio file in, so take the first entry.
            word_stamps = raw_stamps[0] if isinstance(raw_stamps, list) else []

        segments = _group_word_stamps(word_stamps)

    return {"text": text, "segments": segments}


def transcribe_file(audio_path) -> str:
    """Original behavior, unchanged: returns the flat transcript text only.
    Used wherever timestamps aren't needed."""
    return _run_transcription(audio_path, with_timestamps=False)["text"]


def transcribe_file_with_timestamps(audio_path) -> dict:
    """Returns {"text": str, "segments": [{"start", "end", "text"}, ...]}.

    `segments` will be an empty list if the forced aligner wasn't available
    at model load time -- callers should treat that as "no timestamps this
    run" rather than an error.
    """
    return _run_transcription(audio_path, with_timestamps=True)
