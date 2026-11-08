from fastapi import APIRouter, HTTPException

from schemas.conversation import (
    ConversationAnalysisRequest,
    ConversationAnalysisResponse,
)
from services import conversation_service

router = APIRouter(prefix="/api/conversation", tags=["Conversation Analysis"])


@router.post("/analyze", response_model=ConversationAnalysisResponse)
def analyze_conversation_endpoint(request: ConversationAnalysisRequest):
    try:
        segments = [
            segment.model_dump()
            for segment in request.segments
        ]

        result = conversation_service.analyze_conversation(
            mode=request.mode,
            segments=segments
        )

        return ConversationAnalysisResponse(
            mode=request.mode,
            analysis=result
        )

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))