# ============================================================
# SENSEA API — Translation + TTS Routes
# ============================================================
#
# POST /api/translation   — IndicTrans2 text translation
# POST /api/tts           — Kokoro text-to-speech (returns WAV bytes)
# GET  /api/languages     — Supported language codes
# ============================================================

import logging

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import Response
from schemas import (
    LanguageInfo,
    LanguagesResponse,
    TranslationRequest,
    TranslationResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Translation & TTS"])


# ── POST /api/translation ────────────────────────────────────

@router.post(
    "/translation",
    response_model=TranslationResponse,
    summary="Translate text using IndicTrans2",
    description=(
        "Translate text between any pair of the 22 supported Indic languages + English. "
        "Language codes follow the IndicTrans2 format, e.g. `eng_Latn`, `hin_Deva`. "
        "Fetch `/api/languages` for the full list."
    ),
)
async def translate(body: TranslationRequest) -> TranslationResponse:
    from services.translation_service import translate_text  # type: ignore

    try:
        translation = translate_text(
            body.text,
            body.source_language,
            body.target_language,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        )
    except Exception as exc:
        logger.exception("Translation error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Translation failed: {type(exc).__name__}",
        )

    return TranslationResponse(
        translation=translation,
        source_language=body.source_language,
        target_language=body.target_language,
    )


# ── POST /api/tts ────────────────────────────────────────────

class _TTSRequest:
    """Simple dataclass for TTS body."""
    pass


from pydantic import BaseModel, Field


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to convert to speech")


@router.post(
    "/tts",
    summary="Text-to-speech using Kokoro",
    description=(
        "Convert text to speech. "
        "Automatically selects the English voice (`bf_emma`) or "
        "Hindi/Devanagari voice (`hf_alpha`) based on the script. "
        "Returns a WAV audio file (24 kHz, mono)."
    ),
    responses={
        200: {
            "content": {"audio/wav": {}},
            "description": "WAV audio file",
        }
    },
)
async def tts(body: TTSRequest) -> Response:
    from services.tts_service import text_to_speech_bytes  # type: ignore

    try:
        wav_bytes = text_to_speech_bytes(body.text)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        )
    except Exception as exc:
        logger.exception("TTS error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"TTS failed: {type(exc).__name__}",
        )

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=speech.wav"},
    )


# ── GET /api/languages ───────────────────────────────────────

@router.get(
    "/languages",
    response_model=LanguagesResponse,
    summary="List supported languages",
    description="Returns all supported IndicTrans2 language codes for transcription and translation.",
)
async def languages() -> LanguagesResponse:
    from services.speech_service import SPEECH_LANGUAGE_CONFIGS  # type: ignore
    from services.translation_service import get_supported_languages  # type: ignore

    speech = [
        LanguageInfo(code=code, name=cfg["name"])
        for code, cfg in SPEECH_LANGUAGE_CONFIGS.items()
    ]
    targets = get_supported_languages()
    translation = [
        LanguageInfo(code=code, name=name)
        for code, name in sorted(targets.items())
    ]
    return LanguagesResponse(speech_languages=speech, translation_targets=translation)
