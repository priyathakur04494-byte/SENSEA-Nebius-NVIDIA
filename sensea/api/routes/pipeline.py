# POST /api/pipeline — diarize + transcribe + (optional) translate in one call
import logging
from typing import Annotated, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from schemas import PipelineResponse, PipelineSegment

from routes.speech import _read_audio

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["Pipeline"])


@router.post(
    "/pipeline",
    response_model=PipelineResponse,
    summary="Full pipeline: speaker diarization + STT + optional translation",
)
def pipeline(
    audio: Annotated[UploadFile, File(description="Audio file")],
    speech_language: Annotated[str, Form(description="eng_Latn | hin_Deva | hinglish")] = "eng_Latn",
    translate_to: Annotated[Optional[str], Form(description="IndicTrans2 code, e.g. hin_Deva")] = None,
) -> PipelineResponse:
    from services.pipeline_service import run_pipeline  # type: ignore

    data = _read_audio(audio)
    translate_to = (translate_to or "").strip() or None

    try:
        result = run_pipeline(data, speech_language, translate_to)
    except FileNotFoundError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc))
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc))
    except TimeoutError as exc:
        raise HTTPException(status.HTTP_504_GATEWAY_TIMEOUT, str(exc))
    except RuntimeError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc))
    except Exception as exc:
        logger.exception("Pipeline error")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"Pipeline failed: {type(exc).__name__}")

    return PipelineResponse(
        segments=[PipelineSegment(**s) for s in result["segments"]],
        total_speakers=result["total_speakers"],
    )
