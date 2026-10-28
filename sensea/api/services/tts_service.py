# ============================================================
# SENSEA API — TTS Service
# ============================================================
#
# Wraps tts.py (Kokoro).
# Returns generated WAV audio as bytes instead of playing locally.
# Pipelines are lazy-loaded and cached per language code.
# ============================================================

from __future__ import annotations

import io
import logging

import numpy as np

logger = logging.getLogger(__name__)

_tts_ready = False


def _ensure_ready() -> None:
    global _tts_ready
    if _tts_ready:
        return
    try:
        import tts  # noqa: F401  # type: ignore
        _tts_ready = True
        logger.info("TTS module (Kokoro) imported successfully.")
    except Exception as exc:
        logger.warning("TTS module unavailable: %s", exc)
        raise RuntimeError(
            f"TTS module (Kokoro) could not be loaded: {exc}. "
            "Ensure kokoro is installed."
        ) from exc


def text_to_speech_bytes(text: str) -> bytes:
    """
    Convert *text* to speech using Kokoro and return WAV audio as bytes.

    Detects Hindi/Devanagari script automatically and uses the
    appropriate Kokoro voice (bf_emma for English, hf_alpha for Hindi).

    Parameters
    ----------
    text : str
        The text to speak.  Must be non-empty.

    Returns
    -------
    bytes
        A complete WAV file (24 kHz, mono, float32 → PCM_16) as bytes.

    Raises
    ------
    ValueError
        If *text* is empty.
    RuntimeError
        If Kokoro is not installed or audio generation fails.
    """
    if not text or not text.strip():
        raise ValueError("TTS input text must not be empty.")

    _ensure_ready()

    from tts import detect_lang_code, get_pipeline, VOICES  # type: ignore
    import soundfile as sf  # type: ignore

    lang_code = detect_lang_code(text)
    voice = VOICES.get(lang_code, VOICES.get("b", "bf_emma"))

    try:
        pipeline = get_pipeline(lang_code)
        generator = pipeline(text, voice=voice)

        chunks = []
        for _, _, audio in generator:
            chunks.append(audio)

        if not chunks:
            raise RuntimeError("Kokoro produced no audio for the given text.")

        full_audio = np.concatenate(chunks)

        # Write to an in-memory WAV buffer (24 kHz, same as Kokoro output)
        buf = io.BytesIO()
        sf.write(buf, full_audio, samplerate=24000, subtype="PCM_16", format="WAV")
        buf.seek(0)
        return buf.read()

    except Exception as exc:
        logger.error("TTS generation failed: %s", type(exc).__name__)
        raise RuntimeError(f"TTS generation failed: {exc}") from exc
