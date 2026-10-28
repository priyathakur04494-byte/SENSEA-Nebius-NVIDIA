# ============================================================
# SENSEA API — Translation Service
# ============================================================
#
# Wraps translate.py (IndicTrans2).
# Models are lazy-loaded on first call, then cached.
# ============================================================

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_translation_ready = False


def _ensure_ready() -> None:
    global _translation_ready
    if _translation_ready:
        return
    try:
        # Importing translate triggers device detection;
        # actual model weights load lazily on first translate call.
        import translate  # noqa: F401  # type: ignore
        _translation_ready = True
        logger.info("Translation module imported successfully.")
    except Exception as exc:
        logger.warning("Translation module unavailable: %s", exc)
        raise RuntimeError(
            f"Translation module could not be loaded: {exc}. "
            "Ensure IndicTrans2 and IndicTransToolkit are installed."
        ) from exc


def translate_text(
    text: str,
    source_language: str,
    target_language: str,
) -> str:
    """
    Translate *text* from *source_language* to *target_language*.

    Both language codes must be valid IndicTrans2 codes
    (e.g. 'eng_Latn', 'hin_Deva').

    Returns the translated string.
    Raises:
        ValueError / UnsupportedLanguageError — bad language code
        RuntimeError — translation module not installed
    """
    _ensure_ready()

    from translate import translate_text as _translate  # type: ignore
    from language_codes import UnsupportedLanguageError  # type: ignore

    if not text.strip():
        return ""

    try:
        return _translate(text, source_language, target_language)
    except UnsupportedLanguageError:
        raise
    except Exception as exc:
        logger.error("Translation failed: %s", type(exc).__name__)
        raise RuntimeError(f"Translation failed: {exc}") from exc


def get_supported_languages() -> dict:
    """Return the dict of supported IndicTrans2 language codes and names."""
    from language_codes import TARGET_LANGUAGES  # type: ignore
    return dict(TARGET_LANGUAGES)
