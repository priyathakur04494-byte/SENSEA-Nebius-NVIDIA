import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit import IndicProcessor


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
# CORE TRANSLATION CALL (shared by all three directions)
# ============================================================

def _run_translation(text: str, src_lang: str, tgt_lang: str, direction: str) -> str:
    tokenizer, model = _load_model(direction)

    batch = [text]
    batch = _processor.preprocess_batch(batch, src_lang=src_lang, tgt_lang=tgt_lang)

    inputs = tokenizer(
        batch,
        truncation=True,
        padding="longest",
        return_tensors="pt",
        return_attention_mask=True,
    )
    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            use_cache=True,
            max_length=256,
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


# ============================================================
# GENERAL TRANSLATION FUNCTION
# ============================================================
#
# This is the only function transcribe.py calls:
#   translator(transcript, source_language_code, target_language_code)
# Both codes are full IndicTrans2 codes, e.g. "eng_Latn", "hin_Deva".

def translate_text(text: str, source_language: str, target_language: str) -> str:
    if not text or not text.strip():
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
    print("TEST COMPLETE")
    print("========================================")