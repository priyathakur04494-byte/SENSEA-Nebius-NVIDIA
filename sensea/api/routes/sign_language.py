# ============================================================
# SENSEA API — Sign Language Routes
# ============================================================
#
# POST /api/sign-language/predict — from video upload
# POST /api/sign-language/predict-frames — from raw feature frames (JSON)
# ============================================================

import logging
from typing import Annotated, List

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field
from schemas import SignPredictResponse, SignPrediction

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/sign-language", tags=["Sign Language"])

_MAX_VIDEO_BYTES = 100 * 1024 * 1024  # 100 MB

_VIDEO_CONTENT_TYPES = {
    "video/mp4", "video/quicktime", "video/webm",
    "video/x-msvideo", "video/x-matroska",
    "application/octet-stream",
}


# ── POST /api/sign-language/predict ─────────────────────────

@router.post(
    "/predict",
    response_model=SignPredictResponse,
    summary="Predict sign language from video",
    description=(
        "Upload a short video clip (MP4, MOV, WebM). "
        "The server extracts 30 frames using MediaPipe (hand + pose landmarks), "
        "runs orientation detection, then classifies the sign with the Conv1D+BiGRU model. "
        "Supported signs: hello, goodbye, thank_you, sorry, please, yes, no, help, "
        "stop, okay, me, you, mother, father, friend, home, hospital, eat, drink, water."
    ),
)
async def predict_from_video(
    video: Annotated[UploadFile, File(description="Short video clip of a sign language gesture")],
) -> SignPredictResponse:
    from services.sign_language_service import predict_from_video as _predict  # type: ignore

    data = video.file.read()
    if not data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded video file is empty.",
        )
    if len(data) > _MAX_VIDEO_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Video too large (max {_MAX_VIDEO_BYTES // 1024 // 1024} MB).",
        )

    try:
        result = _predict(data)
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
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        )
    except Exception as exc:
        logger.exception("Sign language prediction error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {type(exc).__name__}",
        )

    return SignPredictResponse(
        prediction=SignPrediction(**result["prediction"]),
        frames_used=result["frames_used"],
        orientation=result["orientation"],
    )


# ── POST /api/sign-language/predict-frames ──────────────────

class FramesRequest(BaseModel):
    frames: List[List[float]] = Field(
        ...,
        description="List of up to 30 feature vectors, each of length 258."
    )
    orientation: str = Field(
        default="original",
        description="'original' or 'flipped' — used for documentation only."
    )


@router.post(
    "/predict-frames",
    response_model=SignPredictResponse,
    summary="Predict sign language from pre-extracted feature frames",
    description=(
        "Send pre-extracted MediaPipe feature frames as JSON (useful when the "
        "client does landmark extraction locally, e.g. on-device MediaPipe). "
        "Each frame must be a list of 258 floats."
    ),
)
async def predict_from_frames(body: FramesRequest) -> SignPredictResponse:
    from services.sign_language_service import predict_from_frames as _predict  # type: ignore

    if not body.frames:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="frames must not be empty.",
        )

    try:
        result = _predict(body.frames, orientation=body.orientation)
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
        logger.exception("Sign language frames prediction error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {type(exc).__name__}",
        )

    return SignPredictResponse(
        prediction=SignPrediction(**result["prediction"]),
        frames_used=result["frames_used"],
        orientation=result["orientation"],
    )
