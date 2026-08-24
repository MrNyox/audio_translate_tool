import threading
from pathlib import Path

import config

_lock = threading.Lock()
_inference_lock = threading.Lock()

_pipe = None
_device = None


def is_model_loaded() -> bool:
    return _pipe is not None


def _resolve_device(torch_module):
    requested = (config.DEVICE or "auto").lower()

    if requested == "auto":
        if torch_module.cuda.is_available():
            return "cuda"

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

    # Avoid float16 on CPU unless explicitly forced by the user.
    if device == "cpu" and requested in {"", "auto"} and dtype == torch_module.float16:
        return torch_module.float32

    return dtype


def load_model() -> None:
    global _pipe, _device

    with _lock:
        if _pipe is not None:
            return

        try:
            import torch
            from transformers import (
                AutoModelForSpeechSeq2Seq,
                AutoProcessor,
                pipeline,
            )
        except Exception as exc:
            raise RuntimeError(
                "Qwen3-ASR dependencies are missing. Install torch, transformers, and soundfile."
            ) from exc

        device = _resolve_device(torch)
        torch_dtype = _resolve_torch_dtype(torch, device)

        try:
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                config.MODEL_ID,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )

            processor = AutoProcessor.from_pretrained(
                config.MODEL_ID,
                trust_remote_code=True,
            )

            model.to(device)
            model.eval()

            pipeline_kwargs = {
                "model": model,
                "device": device,
                "torch_dtype": torch_dtype,
            }

            tokenizer = getattr(processor, "tokenizer", None)
            feature_extractor = getattr(processor, "feature_extractor", None)

            if tokenizer is not None:
                pipeline_kwargs["tokenizer"] = tokenizer

            if feature_extractor is not None:
                pipeline_kwargs["feature_extractor"] = feature_extractor

            _pipe = pipeline(
                "automatic-speech-recognition",
                **pipeline_kwargs,
            )

            _device = device

        except Exception as exc:
            _pipe = None
            _device = None
            raise RuntimeError(f"Failed to load Qwen3-ASR model: {exc}") from exc


def transcribe_file(audio_path) -> str:
    audio_path = Path(audio_path)

    if not audio_path.is_file():
        raise RuntimeError("Extracted audio file not found.")

    load_model()

    with _inference_lock:
        try:
            result = _pipe(str(audio_path))
        except Exception as exc:
            raise RuntimeError(f"Transcription failed: {exc}") from exc

    if isinstance(result, (list, tuple)):
        result = result[0] if result else {}

    if isinstance(result, dict):
        return str(result.get("text") or "").strip()

    return str(result or "").strip()
