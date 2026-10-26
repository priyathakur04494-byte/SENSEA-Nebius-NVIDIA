import re

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit import IndicProcessor

from language_codes import (
    UnsupportedLanguageError,
    validate_language_code,
)


# ============================================================
# CONFIG
# ============================================================

MODEL_NAMES = {
    "en-indic": "ai4bharat/indictrans2-en-indic-dist-200M",
    "indic-en": "ai4bharat/indictrans2-indic-en-dist-200M",
    "indic-indic": "ai4bharat/indictrans2-indic-indic-dist-320M",
}

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Using device: {DEVICE}")


# ============================================================
# NOTE ON LANGUAGE CODES
# ============================================================
#
# transcribe.py already sends IndicTrans2's native codes directly
# (eng_Latn, hin_Deva, asm_Beng, tam_Taml, ...) — see TARGET_LANGUAGES
# and SPEECH_LANGUAGES in transcribe.py. This file now works with
# those codes AS-IS. There is no short-code -> full-code mapping here
# anymore — that mismatch (LANGUAGES only knew "en"/"hi"/"mr"/"gu"/
# "pa"/"bn") was the cause of "Unsupported source language: eng_Latn".
#
# The codes are validated in Python before they reach the toolkit
# (see "LANGUAGE CODE VALIDATION" below and language_codes.py).


# ============================================================
# LANGUAGE CODE VALIDATION
# ============================================================
#
# IndicTransToolkit's native processor reads the script part of a
# code with an UNCHECKED list index:
#
#     cdef str script_part = src_lang.split("_")[1]     # processor.pyx:380
#
# A code WITHOUT an underscore ("hin", "mar", "klingon") makes
# split("_") return a 1-element list, so index 1 reads past the end
# of the list and the C extension INCREFs a garbage pointer. The
# process dies with SIGSEGV inside processor.cpython-*.so — no Python
# exception, nothing to catch, the interpreter is simply gone.
#
# It needs no model, no MPS/Metal and no tokenizer: preprocess_batch()
# alone reproduces it. Validating here keeps that code path
# unreachable.

__all__ = [
    "translate_text",
    "UnsupportedLanguageError",
    "TranslationInputTooLongError",
]


class TranslationInputTooLongError(ValueError):
    """Raised when a chunk still exceeds the encoder token limit.

    With the chunking below this should never happen - it exists so
    that an over-long input fails loudly instead of being silently
    truncated.
    """


# ============================================================
# LONG INPUT HANDLING
# ============================================================
#
# IndicTrans2's encoder is trained on 256 source positions (and its
# tokenizer advertises model_max_length = 256). Two things used to
# lose content on long passages:
#
#   1. tokenizer(..., truncation=True) silently dropped everything
#      past the 256 token encoder limit,
#   2. generate(..., max_length=256) capped the OUTPUT at 256
#      tokens. IndicTrans2 emits roughly one token per akshara, so an
#      Indic output needs about TWICE as many tokens as its English
#      source (measured: 24 source tokens -> 37..48 output tokens for
#      hi/ta/ur/ml). A 256 token cap therefore cut the tail off most
#      Indic translations.
#
# Long text is now split into chunks that fit the encoder and
# translated chunk by chunk, and the output budget of each chunk
# scales with its input. The decoder uses sinusoidal position
# embeddings (modeling_indictrans.py: "produces sinusoidal positional
# embeddings of any length"), so a longer output is safe.
#
# Text that already fits is passed through untouched, so short inputs
# behave exactly as before.

ENCODER_TOKEN_LIMIT = 256
CHUNK_TOKEN_BUDGET = 240          # 256 minus room for the language tags
MIN_OUTPUT_TOKENS = 256           # the old fixed max_length
OUTPUT_TOKEN_MULTIPLIER = 2       # worst observed source -> output ratio
OUTPUT_TOKEN_HEADROOM = 32
MAX_OUTPUT_TOKENS = 1024          # safety cap

_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?।\n])\s+")


# ============================================================
# LAZY MODEL LOADING
# ============================================================
#
# Three checkpoints exist for three directions. We only load the one(s)
# actually used in a session — loading all three up front means ~3x the
# download + ~3x the RAM for models you might not need this run.

_loaded_models: dict[str, tuple] = {}
_processor = IndicProcessor(inference=True)  # one processor works for every direction


def _load_model(direction: str):
    if direction in _loaded_models:
        return _loaded_models[direction]

    model_name = MODEL_NAMES[direction]
    print(f"\nLoading {direction} model ({model_name})...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name, trust_remote_code=True
    ).to(DEVICE)
    model.eval()

    _loaded_models[direction] = (tokenizer, model)
    print(f"{direction} model loaded!")
    return tokenizer, model


def _get_direction(source_lang: str, target_lang: str) -> str:
    source_is_english = source_lang == "eng_Latn"
    target_is_english = target_lang == "eng_Latn"

    if source_is_english and target_is_english:
        raise ValueError("Source and target are both English — nothing to translate")
    if source_is_english:
        return "en-indic"
    if target_is_english:
        return "indic-en"
    return "indic-indic"


# ============================================================
# CHUNKING HELPERS
# ============================================================

def _encoder_limit(tokenizer) -> int:
    """Token limit of the model encoder (512 for IndicTrans2)."""
    limit = getattr(tokenizer, "model_max_length", ENCODER_TOKEN_LIMIT)
    if not isinstance(limit, int) or limit <= 0 or limit > 10000:
        limit = ENCODER_TOKEN_LIMIT
    return min(limit, ENCODER_TOKEN_LIMIT)


def _count_tokens(tokenizer, text: str) -> int:
    """Number of source tokens in `text`.

    The IndicTrans2 tokenizer's own encode() expects the
    "<src> <tgt> <text>" form and ASSERTS on the language tags, so it
    cannot be used on raw text. The underlying SentencePiece model can.
    """
    spm = getattr(tokenizer, "src_spm", None)
    if spm is not None:
        try:
            return len(spm.EncodeAsPieces(text))
        except Exception:
            pass
    return max(1, len(text.split()))


def _split_sentences(text: str) -> list[str]:
    parts = [part.strip() for part in _SENTENCE_BOUNDARY_RE.split(text) if part.strip()]
    return parts or [text.strip()]


def _split_by_budget(text: str, tokenizer, budget: int) -> list[str]:
    """Split one oversized sentence into word groups that fit."""
    words = text.split()
    pieces: list[str] = []
    current: list[str] = []
    current_len = 0

    for word in words:
        word_len = _count_tokens(tokenizer, word)
        if current and current_len + word_len > budget:
            pieces.append(" ".join(current))
            current = []
            current_len = 0
        current.append(word)
        current_len += word_len

    if current:
        pieces.append(" ".join(current))

    return pieces or [text]


def _split_into_chunks(text: str, tokenizer) -> list[str]:
    """Split `text` into chunks that fit the encoder.

    Text that already fits is returned as a single untouched chunk,
    so short inputs keep their exact previous behaviour.
    """
    limit = _encoder_limit(tokenizer)
    budget = min(CHUNK_TOKEN_BUDGET, limit - 16)

    if _count_tokens(tokenizer, text) <= budget:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in _split_sentences(text):
        sentence_len = _count_tokens(tokenizer, sentence)

        if sentence_len > budget:
            if current:
                chunks.append(" ".join(current))
                current = []
                current_len = 0
            chunks.extend(_split_by_budget(sentence, tokenizer, budget))
            continue

        if current and current_len + sentence_len > budget:
            chunks.append(" ".join(current))
            current = []
            current_len = 0

        current.append(sentence)
        current_len += sentence_len

    if current:
        chunks.append(" ".join(current))

    return chunks


# ============================================================
# CORE TRANSLATION CALL (shared by all three directions)
# ============================================================

def _translate_chunk(text, src_lang, tgt_lang, tokenizer, model) -> str:
    batch = _processor.preprocess_batch(
        [text], src_lang=src_lang, tgt_lang=tgt_lang
    )

    inputs = tokenizer(
        batch,
        truncation=False,
        padding="longest",
        return_tensors="pt",
        return_attention_mask=True,
    )

    limit = _encoder_limit(tokenizer)
    input_tokens = inputs["input_ids"].shape[1]

    if input_tokens > limit:
        raise TranslationInputTooLongError(
            f"Chunk of {input_tokens} tokens exceeds the {limit} token "
            f"encoder limit"
        )

    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

    # IndicTrans2 needs roughly twice as many output tokens as input
    # tokens (one token per akshara), so the output budget scales with
    # the chunk instead of being a flat 256.
    max_length = min(
        MAX_OUTPUT_TOKENS,
        max(
            MIN_OUTPUT_TOKENS,
            input_tokens * OUTPUT_TOKEN_MULTIPLIER + OUTPUT_TOKEN_HEADROOM,
        ),
    )

    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            use_cache=True,
            max_length=max_length,
            num_beams=5,
            num_return_sequences=1,
        )

    decoded = tokenizer.batch_decode(
        generated_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    translations = _processor.postprocess_batch(decoded, lang=tgt_lang)
    return translations[0].strip()


def _run_translation(text: str, src_lang: str, tgt_lang: str, direction: str) -> str:
    tokenizer, model = _load_model(direction)

    translations = []

    for chunk in _split_into_chunks(text, tokenizer):
        translations.append(
            _translate_chunk(chunk, src_lang, tgt_lang, tokenizer, model)
        )

    return " ".join(part for part in translations if part).strip()


# ============================================================
# GENERAL TRANSLATION FUNCTION
# ============================================================
#
# This is the only function transcribe.py calls:
#   translator(transcript, source_language_code, target_language_code)
# Both codes are full IndicTrans2 codes, e.g. "eng_Latn", "hin_Deva".
#
# The codes are validated BEFORE anything else happens, so a malformed
# code raises a clear UnsupportedLanguageError instead of taking the
# process down with a native SIGSEGV.

def translate_text(text: str, source_language: str, target_language: str) -> str:
    if not isinstance(text, str):
        raise TypeError(
            f"Cannot translate {type(text).__name__}: expected a string"
        )

    source_language = validate_language_code(source_language, "source language")
    target_language = validate_language_code(target_language, "target language")

    if not text.strip():
        return ""

    if source_language == target_language:
        return text

    direction = _get_direction(source_language, target_language)
    return _run_translation(text, source_language, target_language, direction)


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("\n========================================")
    print("ENGLISH -> INDIC TEST")
    print("========================================")

    english_text = "I got some very good news today."
    en_targets = [
        ("hin_Deva", "Hindi"),
        ("mar_Deva", "Marathi"),
        ("guj_Gujr", "Gujarati"),
        ("pan_Guru", "Punjabi"),
        ("ben_Beng", "Bengali"),
        ("asm_Beng", "Assamese"),
        ("tam_Taml", "Tamil"),
    ]

    print("\nInput:", english_text)
    for code, name in en_targets:
        print(f"\n{name}:")
        try:
            print(translate_text(english_text, "eng_Latn", code))
        except Exception as e:
            print("ERROR:", e)

    print("\n========================================")
    print("INDIC -> ENGLISH TEST")
    print("========================================")

    hindi_text = "मुझे आज बहुत अच्छी खबर मिली।"
    print("\nInput:", hindi_text)
    try:
        print("English:", translate_text(hindi_text, "hin_Deva", "eng_Latn"))
    except Exception as e:
        print("ERROR:", e)

    print("\n========================================")
    print("INDIC -> INDIC TEST")
    print("========================================")

    print("\nInput (Hindi):", hindi_text)
    try:
        print("Tamil:", translate_text(hindi_text, "hin_Deva", "tam_Taml"))
    except Exception as e:
        print("ERROR:", e)

    print("\n========================================")
    print("INVALID LANGUAGE CODE TEST")
    print("========================================")

    for bad_code in ["hin", "mar", "klingon", "xyz_Xyzw", ""]:
        try:
            print(bad_code, "->", translate_text("hello", bad_code, "eng_Latn"))
        except Exception as e:
            print(bad_code, "-> ERROR:", e)

    print("\n========================================")
    print("TEST COMPLETE")
    print("========================================")
