LANGUAGE_CODE_MAP = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "bn": "ben_Beng",
    "mr": "mar_Deva",
    "gu": "guj_Gujr",
    "kn": "kan_Knda",
    "ml": "mal_Mlym",
    "pa": "pan_Guru",
}


def translate_text(text, source_language, target_language):
    if source_language not in LANGUAGE_CODE_MAP:
        raise ValueError(f"Unsupported source language: {source_language}")

    if target_language not in LANGUAGE_CODE_MAP:
        raise ValueError(f"Unsupported target language: {target_language}")

    from translate import translate_text as ml_translate_text

    source_code = LANGUAGE_CODE_MAP[source_language]
    target_code = LANGUAGE_CODE_MAP[target_language]

    return ml_translate_text(
        text,
        source_code,
        target_code
    )