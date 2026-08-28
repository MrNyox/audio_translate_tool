"""Stage Two: LLM-based translation using unsloth/Qwen2.5-7B-Instruct-bnb-4bit.

This model ships PRE-QUANTIZED (bitsandbytes NF4). Do NOT pass a new
BitsAndBytesConfig -- transformers reads the quantization config from the
model's own config.json on Hugging Face Hub.
"""

import gc
import json
import logging
import re
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
    (0x0400, 0x04FF),    # Cyrillic
    (0x0500, 0x052F),    # Cyrillic Supplement
    (0x0370, 0x03FF),    # Greek and Coptic
    (0x0590, 0x05FF),    # Hebrew
    (0x0E00, 0x0E7F),    # Thai
    (0x0900, 0x097F),    # Devanagari
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


def _build_batch_system_prompt(target_language: str, count: int) -> str:
    return (
        f"You are an expert translator. Translate the following list of text segments "
        f"into {target_language}. Write the ENTIRE translation in {target_language}, "
        f"using its native script throughout. Proper nouns, brand names, and URLs may "
        f"stay in their original Latin spelling, but every other word must be in "
        f"{target_language}. Never switch into Chinese, Japanese, Korean, or any other "
        f"unrelated language partway through.\n\n"
        f"Output ONLY a valid JSON array of strings. Each string in the array must be "
        f"the translation of the corresponding input segment in the exact same order. "
        f"Do not output markdown blocks like ```json. Do not include any explanations, "
        f"notes, or conversational filler. Just the raw JSON array.\n"
        f"Prioritize natural, fluent, idiomatic phrasing while preserving the original "
        f"meaning, tone, and context."
    )


def _parse_translated_json_array(translated_raw: str) -> List[str]:
    """Best-effort extraction of a JSON string array from raw LLM output."""
    translated_list: List[str] = []
    try:
        match = re.search(r'\[.*', translated_raw, re.DOTALL)
        if match:
            json_str = match.group(0)

            # Append closing bracket if generation was cut off
            if not json_str.rstrip().endswith(']'):
                json_str += ']'

            try:
                parsed = json.loads(json_str)
                if isinstance(parsed, list):
                    translated_list = [str(item).strip() for item in parsed]
                else:
                    # Fallback if it returned a dict: find the list inside the values
                    for v in parsed.values():
                        if isinstance(v, list):
                            translated_list = [str(item).strip() for item in v]
                            break
            except json.JSONDecodeError:
                # Fallback for malformed arrays (e.g., missing commas): extract quoted strings
                strings = re.findall(r'"((?:[^"\\]|\\.)*)"', json_str)
                translated_list = [
                    s.encode().decode("unicode_escape").strip() for s in strings
                ]
        else:
            # Fallback if no brackets are found at all
            translated_list = [
                line.strip() for line in translated_raw.split("\n") if line.strip()
            ]
    except Exception as exc:
        logger.warning(
            "Failed to parse JSON array from LLM output, falling back to line split: %s",
            exc,
        )
        translated_list = [
            line.strip() for line in translated_raw.split("\n") if line.strip()
        ]
    return translated_list


def _generate_batch_translation(source_texts: List[str], target_language: str) -> List[str]:
    """Run one batched JSON-array translation call. Assumes caller holds `_lock`."""
    system_prompt = _build_batch_system_prompt(target_language, len(source_texts))
    user_content = json.dumps(source_texts, ensure_ascii=False)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
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
        max_length=config.TRANSLATION_MAX_TOKENS + 512,
    ).to(_model.device)

    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=config.TRANSLATION_MAX_TOKENS,
            do_sample=False,
            repetition_penalty=1.1,
            suppress_tokens=_get_suppressed_token_ids() or None,
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    translated_raw = _tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    del inputs, outputs, generated_tokens

    return _parse_translated_json_array(translated_raw)


def _redistribute_batch_timestamps(batch: List[dict], translated_list: List[str]) -> List[tuple]:
    """
    Used only when the model returns a different number of translated
    strings than input segments (it merged or split fragments while
    translating -- common because the input segments are fixed-length ASR
    chunks, not sentence boundaries, and a fluent translation naturally
    wants to recombine them).

    There's no way to know exactly which original segments a given merged
    translated string corresponds to without re-running alignment, so
    instead of guessing (or discarding the mismatch and losing content),
    this spreads the batch's overall time window -- first segment's start
    to last segment's end -- across the translated strings proportionally
    to each string's length. This keeps the model's original, fully
    context-aware translation intact (no quality loss from re-translating
    fragments in isolation) and guarantees every translated string gets a
    reasonable, gap-free window and nothing is dropped. The exact cue
    boundaries here don't need to be precise: `normalize_subtitle_pacing`
    re-times everything to actual reading speed afterward anyway.
    """
    batch_start = float(batch[0].get("start", 0.0) or 0.0)
    batch_end = float(batch[-1].get("end", batch_start) or batch_start)
    if batch_end <= batch_start:
        batch_end = batch_start + max(len(translated_list), 1) * 1.0

    total_span = batch_end - batch_start
    weights = [max(1, len(t.split())) for t in translated_list]
    total_weight = sum(weights) or 1

    result = []
    cursor = batch_start
    for w in weights:
        duration = total_span * (w / total_weight)
        seg_start = cursor
        seg_end = cursor + duration
        result.append((round(seg_start, 3), round(seg_end, 3)))
        cursor = seg_end

    if result:
        # Snap the last cue's end to the batch's true end to avoid drift
        # from repeated float addition.
        last_start, _ = result[-1]
        result[-1] = (last_start, round(batch_end, 3))

    return result


def translate_segments(segments: List[dict], target_language: str) -> List[dict]:
    """Translate a list of {"start", "end", "text"} segments.

    Groups segments into large batches (up to the token budget) to give
    the model enough context to prevent language drift and hallucinations
    (stray words from other languages). It then prompts the model to output
    a JSON array of translated strings, mapping 1-to-1 back to the input
    segments when possible.

    That 1-to-1 mapping isn't guaranteed just by asking for it: the input
    segments are fixed-length ASR chunks that often cut mid-sentence, and a
    fluent translation naturally wants to merge such fragments into fewer,
    more complete sentences. When that happens, the returned list is
    shorter (or longer) than the input batch, so index-based timestamp
    assignment would misattribute text to the wrong window and silently
    drop the rest. When the counts match, each translated string keeps its
    corresponding segment's exact original timestamps. When they don't,
    `_redistribute_batch_timestamps` spreads the batch's overall time
    span across the translated strings instead, so nothing is lost.
    """
    with _lock:
        load_translation_model()

        translated_segments: List[dict] = []

        effective_max = config.TRANSLATION_CHUNK_SIZE - 256
        current_batch_segments: List[dict] = []
        current_tokens = 0
        batches = []

        # 1. Group segments into batches that fit the context window (similar to how translate_text works)
        for seg in segments:
            source_text = str(seg.get("text", "")).strip()
            line_tokens = len(_tokenizer.encode(source_text, add_special_tokens=False))

            if current_tokens + line_tokens > effective_max and current_batch_segments:
                batches.append(current_batch_segments)
                current_batch_segments = []
                current_tokens = 0

            current_batch_segments.append(seg)
            current_tokens += line_tokens

        if current_batch_segments:
            batches.append(current_batch_segments)

        # 2. Translate each batch
        for batch in batches:
            source_texts = [str(seg.get("text", "")).strip() for seg in batch]

            translated_list = _generate_batch_translation(source_texts, target_language)

            if not translated_list:
                # Nothing usable came back at all; skip rather than fabricate content.
                continue

            if len(translated_list) == len(batch):
                # Clean 1:1 mapping -- each translated string keeps its
                # corresponding segment's exact original timestamps.
                for seg, text in zip(batch, translated_list):
                    translated_segments.append(
                        {"start": seg.get("start"), "end": seg.get("end"), "text": text}
                    )
            else:
                logger.info(
                    "Batch translation returned %d items for %d input segments "
                    "(model merged/split fragments during translation); "
                    "redistributing timestamps proportionally across the "
                    "batch's time window instead of dropping content.",
                    len(translated_list), len(batch),
                )
                windows = _redistribute_batch_timestamps(batch, translated_list)
                for (start, end), text in zip(windows, translated_list):
                    translated_segments.append({"start": start, "end": end, "text": text})

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
