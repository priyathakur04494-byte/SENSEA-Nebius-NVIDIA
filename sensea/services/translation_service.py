# ============================================================
# SENSEA API — Translation Service
# ============================================================
#
# Primary engine: translate.py (IndicTrans2)
# Fallback 1: NVIDIA Nemotron LLM (when NVIDIA_API_KEY is configured)
# Fallback 2: Free web translation API (for instant dev without keys)
# ============================================================

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from typing import Optional

logger = logging.getLogger(__name__)

_indictrans_available: Optional[bool] = None

INDIC_TO_ISO = {
    "eng_Latn": "en",
    "hin_Deva": "hi",
    "tam_Taml": "ta",
    "tel_Telu": "te",
    "ben_Beng": "bn",
    "mar_Deva": "mr",
    "guj_Gujr": "gu",
    "kan_Knda": "kn",
    "mal_Mlym": "ml",
    "pan_Guru": "pa",
}


def _check_indictrans() -> bool:
    global _indictrans_available
    if _indictrans_available is not None:
        return _indictrans_available
    try:
        import translate  # noqa: F401  # type: ignore
        _indictrans_available = True
        logger.info("IndicTrans2 translation module ready.")
    except Exception as exc:
        _indictrans_available = False
        logger.info(
            "IndicTrans2 not available (%s). Will use fallback engines.",
            exc,
        )
    return _indictrans_available


def _translate_free_web(text: str, source_language: str, target_language: str) -> str:
    """Free web translation fallback that requires no API keys or compilers."""
    sl = INDIC_TO_ISO.get(source_language, "en")
    tl = INDIC_TO_ISO.get(target_language, "hi")
    q = urllib.parse.quote(text)
    url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl={sl}&tl={tl}&dt=t&q={q}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=5) as res:
        data = json.loads(res.read().decode("utf-8"))
        if data and isinstance(data[0], list):
            translated = "".join(part[0] for part in data[0] if part and part[0])
            if translated.strip():
                return translated.strip()
    raise RuntimeError("Free translation returned empty response")


def _translate_with_nvidia(text: str, source_language: str, target_language: str) -> str:
    """Fallback translation using NVIDIA Nemotron API."""
    import config
    from language_codes import TARGET_LANGUAGES  # type: ignore

    api_key = config.NVIDIA_API_KEY
    if not api_key:
        # Fall back to free web translation
        return _translate_free_web(text, source_language, target_language)

    try:
        from openai import OpenAI

        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key,
            timeout=20.0,
        )

        src_name = TARGET_LANGUAGES.get(source_language, source_language)
        tgt_name = TARGET_LANGUAGES.get(target_language, target_language)

        prompt = (
            f"You are an expert multilingual translator for assistive technology.\n"
            f"Translate the following text from {src_name} to {tgt_name}.\n"
            f"Rules:\n"
            f"1. Return ONLY the translated sentence.\n"
            f"2. Do NOT add notes, explanations, phonetic transliterations, or markdown tags.\n\n"
            f"Text to translate:\n{text}"
        )

        response = client.chat.completions.create(
            model="nvidia/nemotron-3.5-lightning-30b-a3b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=600,
        )

        content = response.choices[0].message.content or ""
        content = content.replace("<think>", "").replace("</think>", "")
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("\n"):
                content = content[1:]
        return content.strip().strip('"')

    except Exception as exc:
        logger.warning("NVIDIA translation failed (%s), trying free web fallback", exc)
        return _translate_free_web(text, source_language, target_language)


def translate_text(
    text: str,
    source_language: str,
    target_language: str,
) -> str:
    """
    Translate *text* from *source_language* to *target_language*.

    Both language codes must be valid IndicTrans2 codes
    (e.g. 'eng_Latn', 'hin_Deva').
    """
    if not text.strip():
        return ""

    if source_language == target_language:
        return text

    from language_codes import validate_language_code, UnsupportedLanguageError  # type: ignore

    try:
        validate_language_code(source_language, is_target=False)
        validate_language_code(target_language, is_target=True)
    except UnsupportedLanguageError:
        raise

    # 1. Try local IndicTrans2 if available
    if _check_indictrans():
        try:
            from translate import translate_text as _translate  # type: ignore
            return _translate(text, source_language, target_language)
        except Exception as exc:
            logger.warning("IndicTrans2 translation failed (%s), falling back", exc)

    # 2. Try NVIDIA Nemotron or Free Web fallback
    try:
        return _translate_with_nvidia(text, source_language, target_language)
    except Exception as exc:
        logger.error("All translation engines failed: %s", exc)
        raise RuntimeError(f"Translation failed: {exc}") from exc


def get_supported_languages() -> dict:
    """Return the dict of supported IndicTrans2 language codes and names."""
    from language_codes import TARGET_LANGUAGES  # type: ignore
    return dict(TARGET_LANGUAGES)
