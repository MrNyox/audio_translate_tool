"""Stage Two: LLM-based translation using unsloth/Qwen2.5-7B-Instruct-bnb-4bit.

This model ships PRE-QUANTIZED (bitsandbytes NF4). Do NOT pass a new
BitsAndBytesConfig -- transformers reads the quantization config from the
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

# --- Language-drift guard --------------------------------------------------
# The base model (Qwen2.5-7B-Instruct) is heavily EN/ZH-trained. Under
# greedy decoding it will occasionally code-switch mid-sentence into
# Chinese even when explicitly asked for Arabic or English -- e.g.
# "أهلاً各位،我是..." instead of a fully Arabic sentence. Since the app only
# ever supports Arabic or English as translation targets (see routes.py),
# Chinese/Japanese/Korean characters should NEVER legitimately appear in
# output. Rather than relying on the prompt alone, we hard-block every
# vocabulary token that decodes to a CJK character via `suppress_tokens`,
# computed once per loaded tokenizer and cached.
_DISALLOWED_SCRIPT_RANGES = [
    (0x4E00, 0x9FFF),    # CJK Unified Ideographs
    (0x3400, 0x4DBF),    # CJK Unified Ideographs Extension A
    (0x3040, 0x309F),    # Hiragana
    (0x30A0, 0x30FF),    # Katakana
    (0xAC00, 0xD7A3),    # Hangul syllables
    (0xF900, 0xFAFF),    # CJK Compatibility Ideographs
]

_suppressed_token_ids_cache = None


def _contains_disallowed_script(text: str) -> bool:
    for ch in text:
        cp = ord(ch)
        for start, end in _DISALLOWED_SCRIPT_RANGES:
            if start <= cp <= end:
                return True
    return False


def _get_suppressed_token_ids():
    """Vocabulary token ids that decode to CJK characters (cached)."""
    global _suppressed_token_ids_cache

    if _suppressed_token_ids_cache is not None:
        return _suppressed_token_ids_cache

    if _tokenizer is None:
        return []

    vocab = _tokenizer.get_vocab()
    suppressed = set()

    # 1. Suppress tokens that directly decode to CJK characters.
    #    `token_str` from vocab.items() is the raw BPE byte-mapped string
    #    (e.g. GPT-2 style unicode), NOT the actual character. We must
    #    decode it to check for CJK.
    for token_str, token_id in vocab.items():
        try:
            decoded = _tokenizer.decode([token_id])
            if _contains_disallowed_script(decoded):
                suppressed.add(token_id)
        except Exception:
            pass

    # 2. Suppress byte-level tokens used to build CJK characters.
    #    Qwen uses byte-level BPE, meaning a CJK character may be split
    #    into multiple byte tokens. We encode a sample of CJK characters
    #    to capture and ban all the subword byte tokens the tokenizer
    #    uses to form them, physically preventing Chinese generation.
    try:
        cjk_sample = "".join(
            chr(cp)
            for start, end in _DISALLOWED_SCRIPT_RANGES
            for cp in range(start, min(start + 200, end + 1))
        )
        cjk_token_ids = _tokenizer.encode(cjk_sample, add_special_tokens=False)
        suppressed.update(cjk_token_ids)

        cjk_with_spaces = " " + " ".join(list(cjk_sample))
        cjk_space_token_ids = _tokenizer.encode(cjk_with_spaces, add_special_tokens=False)
        suppressed.update(cjk_space_token_ids)
    except Exception:
        pass

    _suppressed_token_ids_cache = list(suppressed)
    logger.info(
        "Translation language-drift guard: suppressing %d CJK-containing/byte "
        "tokens during generation.",
        len(_suppressed_token_ids_cache),
    )
    return _suppressed_token_ids_cache


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
    global _model, _tokenizer, _suppressed_token_ids_cache
    with _lock:
        _model = None
        _tokenizer = None
        _suppressed_token_ids_cache = None
    _free_vram()


_SYSTEM_PROMPT_TEMPLATE = (
    "You are an expert translator. Translate the following text into {target_language}. "
    "Write the ENTIRE translation in {target_language}, using its native script throughout. "
    "Proper nouns, brand names, and URLs (e.g. company or website names) may stay in their "
    "original Latin spelling, but every other word must be in {target_language} -- never "
    "switch into Chinese, Japanese, Korean, or any other unrelated language partway through. "
    "Output ONLY the translated text: no explanations, notes, or conversational filler. "
    "Prioritize natural, fluent, idiomatic phrasing while preserving the original meaning, "
    "tone, and context."
)


def _translate_chunk(text: str, target_language: str) -> str:
    """Run a single translation generation call.

    Assumes the model/tokenizer are already loaded and the caller holds
    `_lock`. Shared by both `translate_text` (flat-file path) and
    `translate_segments` (timestamped path) so both stay in sync with any
    future prompt/generation tuning.
    """
    if not text.strip():
        return ""

    system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(target_language=target_language)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
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
            do_sample=False,  # greedy -- deterministic translation
            repetition_penalty=1.1,
            suppress_tokens=_get_suppressed_token_ids() or None,
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    translated = _tokenizer.decode(generated_tokens, skip_special_tokens=True)

    del inputs, outputs, generated_tokens

    translated = translated.strip()

    if _contains_disallowed_script(translated):
        # Should be effectively impossible now that CJK tokens are
        # suppressed at the vocabulary level, but log loudly if it ever
        # happens so it doesn't silently ship bad output again.
        logger.warning(
            "Translated chunk still contains unexpected CJK characters "
            "despite token suppression: %r",
            translated[:120],
        )

    return translated


def translate_text(text: str, target_language: str) -> str:
    """Translate text using greedy decoding. Model persists across calls.

    Unchanged behavior: still chunks by token budget via `_chunk_text` and
    produces the same flat translated.txt output as before.
    """
    with _lock:
        load_translation_model()

        if not text.strip():
            return ""

        chunks = _chunk_text(text, config.TRANSLATION_CHUNK_SIZE)
        translated_chunks: List[str] = [
            _translate_chunk(chunk, target_language) for chunk in chunks
        ]

        return "\n".join(translated_chunks)


def translate_segments(segments: List[dict], target_language: str) -> List[dict]:
    """Translate a list of {"start", "end", "text"} segments one at a time.

    Unlike `translate_text` (which is free to merge/re-split lines across
    an arbitrary token-budget chunk), this translates each segment
    independently so `start`/`end` from ts_transcript.json stay perfectly
    aligned with the translated text in ts_translated.json -- required for
    correct subtitle timing.
    """
    with _lock:
        load_translation_model()

        translated_segments: List[dict] = []
        for seg in segments:
            source_text = str(seg.get("text", "")).strip()
            translated_text = (
                _translate_chunk(source_text, target_language) if source_text else ""
            )
            translated_segments.append(
                {
                    "start": seg.get("start"),
                    "end": seg.get("end"),
                    "text": translated_text,
                }
            )

        return translated_segments


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
