# ============================================================
# SENSEA API — Speech Routes
# ============================================================
#
# POST /api/speech/transcribe   — Whisper.cpp STT
# POST /api/speech/diarize      — Silero VAD + ECAPA-TDNN
# ============================================================

import logging
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from schemas import DiarizeResponse, SpeakerSegment, TranscribeResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/speech", tags=["Speech"])

# Accepted MIME types / extensions for audio uploads
_AUDIO_CONTENT_TYPES = {
    "audio/wav", "audio/wave", "audio/x-wav",
    "audio/webm", "audio/ogg", "audio/flac",
    "audio/mpeg", "audio/mp4",
    "audio/aac", "audio/x-m4a", "audio/m4a",
    "application/octet-stream",   # some clients send this for WAV
}

_MAX_AUDIO_BYTES = 50 * 1024 * 1024  # 50 MB


def _read_audio(upload: UploadFile) -> tuple[bytes, str]:
    """
    Validate and read an uploaded audio file.

    Returns (raw_bytes, original_suffix) where suffix is used by
    speech_service._audio_bytes_to_wav() so ffmpeg receives the right
    file extension hint (e.g. ".m4a" vs ".wav").
    """
    if (
        upload.content_type
        and upload.content_type not in _AUDIO_CONTENT_TYPES
        # Allow content_type=None (Expo sometimes omits it)
    ):
        logger.debug("Unusual content-type: %s", upload.content_type)

    data = upload.file.read()
    if len(data) > _MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Audio file too large (max {_MAX_AUDIO_BYTES // 1024 // 1024} MB).",
        )
    if not data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded audio file is empty.",
        )

    # Derive file extension for ffmpeg format hint
    suffix = ".wav"
    if upload.filename:
        from pathlib import Path as _Path
        ext = _Path(upload.filename).suffix.lower()
        if ext:
            suffix = ext

    return data, suffix


# ── POST /api/speech/transcribe ─────────────────────────────

@router.post(
    "/transcribe",
    response_model=TranscribeResponse,
    summary="Transcribe audio using Whisper.cpp",
    description=(
        "Upload a WAV, FLAC, or OGG file. "
        "m4a/AAC is supported if the server has ffmpeg installed. "
        "Returns the transcript text, detected language name, and Whisper language code. "
        "Supported language codes: `eng_Latn` (English), `hin_Deva` (Hindi), `hinglish`."
    ),
)
async def transcribe(
    audio: Annotated[UploadFile, File(description="Audio file to transcribe")],
    language: Annotated[
        str,
        Form(description="Speech language code: 'eng_Latn', 'hin_Deva', or 'hinglish'"),
    ] = "eng_Latn",
) -> TranscribeResponse:
    from services.speech_service import transcribe_audio  # type: ignore

    audio_bytes, suffix = _read_audio(audio)

    try:
        result = transcribe_audio(audio_bytes, language_code=language, suffix=suffix)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except TimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=str(exc),
        )
    except RuntimeError as exc:
        logger.exception("Transcription runtime error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )

    return TranscribeResponse(**result)


# ── POST /api/speech/diarize ────────────────────────────────

@router.post(
    "/diarize",
    response_model=DiarizeResponse,
    summary="Speaker diarization (who spoke when)",
    description=(
        "Upload a WAV, FLAC, or OGG file. "
        "m4a/AAC is supported if the server has ffmpeg installed. "
        "Returns speaker segments with labels, confidence scores, and duration. "
        "Supports up to 4 speakers per recording."
    ),
)
async def diarize(
    audio: Annotated[UploadFile, File(description="Audio file to diarize")],
) -> DiarizeResponse:
    from services.speech_service import diarize_audio  # type: ignore

    audio_bytes, suffix = _read_audio(audio)

    try:
        result = diarize_audio(audio_bytes, suffix=suffix)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        logger.exception("Diarization error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Diarization failed: {type(exc).__name__}",
        )

    segments = [SpeakerSegment(**s) for s in result["speakers"]]
    return DiarizeResponse(
        speakers=segments,
        total_speakers=result["total_speakers"],
    )
