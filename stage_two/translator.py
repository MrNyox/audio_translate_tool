"""Stage Two: LLM-based translation using unsloth/Qwen2.5-7B-Instruct-bnb-4bit.

This model ships PRE-QUANTIZED (bitsandbytes NF4). Do NOT pass a new
BitsAndBytesConfig — transformers reads the quantization config from the
model's own config.json on Hugging Face Hub.
"""

import gc
import logging
import threading
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import config

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_model = None
_tokenizer = None


def _free_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def unload_previous_model():
    """Free Stage One ASR model from VRAM before loading the LLM."""
    try:
        from stage_one import asr
        # Requires adding an `unload()` function to stage_one/asr.py
        if hasattr(asr, "unload"):
            asr.unload()
    except Exception:
        pass
    _free_vram()


def load_translation_model():
    """Load the pre-quantized translation LLM (idempotent)."""
    global _model, _tokenizer

    if _model is not None and _tokenizer is not None:
        return

    unload_previous_model()

    logger.info("Loading translation model: %s", config.TRANSLATION_MODEL_ID)

    _tokenizer = AutoTokenizer.from_pretrained(
        config.TRANSLATION_MODEL_ID,
        trust_remote_code=False,
    )
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    # NOTE: No quantization_config here. The model is already 4-bit quantized.
    # Passing BitsAndBytesConfig(load_in_4bit=True) would attempt double quantization.
    _model = AutoModelForCausalLM.from_pretrained(
        config.TRANSLATION_MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    _model.eval()
    logger.info("Translation model loaded on device: %s", _model.device)


def unload_translation_model():
    """Explicitly free the translation model (call between stages)."""
    global _model, _tokenizer
    with _lock:
        _model = None
        _tokenizer = None
    _free_vram()


def translate_text(text: str, target_language: str) -> str:
    """Translate text using greedy decoding. Model persists across calls."""
    with _lock:
        load_translation_model()

        if not text.strip():
            return ""

        system_prompt = (
            f"You are an expert translator. Translate the following text to {target_language}. "
            "Output ONLY the translated text. Do not include explanations, notes, or "
            "conversational filler. Prioritize natural, fluent, and idiomatic phrasing "
            "in the target language. Preserve the original meaning, tone, and context."
        )

        chunks = _chunk_text(text, config.TRANSLATION_CHUNK_SIZE)
        translated_chunks: List[str] = []

        for i, chunk in enumerate(chunks):
            logger.debug("Translating chunk %d/%d", i + 1, len(chunks))

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": chunk},
            ]

            input_text = _tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = _tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=config.TRANSLATION_MAX_TOKENS + 512,  # leave room for generation
            ).to(_model.device)

            with torch.no_grad():
                outputs = _model.generate(
                    **inputs,
                    max_new_tokens=config.TRANSLATION_MAX_TOKENS,
                    do_sample=False,  # greedy — deterministic translation
                    repetition_penalty=1.1,
                )

            generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            translated_chunk = _tokenizer.decode(
                generated_tokens, skip_special_tokens=True
            )
            translated_chunks.append(translated_chunk.strip())

            del inputs, outputs, generated_tokens

        return "\n".join(translated_chunks)


def _chunk_text(text: str, max_tokens: int) -> List[str]:
    """Split text into chunks that fit within the model's context window.

    Uses the actual tokenizer for accurate token counting.
    """
    if _tokenizer is None:
        # Fallback if called before model load (shouldn't happen)
        lines = text.split("\n")
        return ["\n".join(lines)]

    lines = text.split("\n")
    chunks: List[str] = []
    current_chunk: List[str] = []
    current_tokens = 0

    # Reserve tokens for system prompt + chat template overhead
    effective_max = max_tokens - 256

    for line in lines:
        line_tokens = len(_tokenizer.encode(line, add_special_tokens=False))

        if current_tokens + line_tokens > effective_max and current_chunk:
            chunks.append("\n".join(current_chunk))
            current_chunk = []
            current_tokens = 0

        current_chunk.append(line)
        current_tokens += line_tokens

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks if chunks else [""]
