import threading
from pathlib import Path
import gc
import config

_lock = threading.Lock()
_inference_lock = threading.Lock()

_model = None
_device = None


def is_model_loaded() -> bool:
    return _model is not None
def unload() -> None:
    """Unload the ASR model and free GPU memory."""
    global _model, _device
    with _lock:
        _model = None
        _device = None
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

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
            return torch_module.bfloat16
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
    """
    global _model, _device

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
            )

            _model = model
            _device = device

        except Exception as exc:
            _model = None
            _device = None
            raise RuntimeError(f"Failed to load Qwen3-ASR model: {exc}") from exc


def transcribe_file(audio_path) -> str:
    audio_path = Path(audio_path)

    if not audio_path.is_file():
        raise RuntimeError("Extracted audio file not found.")

    load_model()

    with _inference_lock:
        try:
            results = _model.transcribe(
                audio=str(audio_path),
                language=config.ASR_LANGUAGE,
            )
        except Exception as exc:
            raise RuntimeError(f"Transcription failed: {exc}") from exc

    if not results:
        return ""

    result = results[0]
    text = getattr(result, "text", None)

    if text is None and isinstance(result, dict):
        text = result.get("text")

    return str(text or "").strip()
# Add to stage_one/asr.py:
def unload() -> None:
    """Free the ASR model from memory."""
    global _model, _device
    with _lock:
        _model = None
        _device = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
