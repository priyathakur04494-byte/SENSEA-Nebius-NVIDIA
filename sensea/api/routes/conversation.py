# ============================================================
# SENSEA API — Conversation Analysis Route
# ============================================================
#
# POST /api/conversation/analyze
# ============================================================

import logging

from fastapi import APIRouter, HTTPException, status
from schemas import AnalyzeRequest, AnalyzeResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/conversation", tags=["Conversation Analysis"])


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Analyze a conversation with NVIDIA Nemotron",
    description=(
        "Send a list of conversation segments to be analyzed by NVIDIA Nemotron. "
        "The `mode` field selects the analysis schema:\n\n"
        "- **general** — summary, key points, important info, questions\n"
        "- **hospital** — medical summary, instructions, medications, tests, appointments\n"
        "- **meeting** — summary, discussion points, decisions, action items, deadlines\n"
        "- **classroom** — summary, topic, key concepts, definitions, flashcards, quiz\n"
    ),
)
async def analyze(body: AnalyzeRequest) -> AnalyzeResponse:
    from services.conversation_service import analyze_conversation  # type: ignore

    segments = [s.model_dump() for s in body.segments]

    try:
        result = analyze_conversation(body.mode, segments)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except RuntimeError as exc:
        logger.exception("Conversation analysis error")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        )
    except Exception as exc:
        logger.exception("Conversation analysis unexpected error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {type(exc).__name__}",
        )

    return AnalyzeResponse(mode=body.mode, result=result)
