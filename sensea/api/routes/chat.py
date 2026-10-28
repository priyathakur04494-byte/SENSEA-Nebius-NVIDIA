# ============================================================
# SENSEA API — Chat / RAG / Memory Routes
# ============================================================
#
# POST   /api/chat           — Answer a question from stored memories
# POST   /api/memories       — Add memories
# GET    /api/memories       — Get memories (filtered by mode/user)
# DELETE /api/memories       — Clear memories
# ============================================================

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, status
from schemas import (
    AddMemoriesRequest,
    AddMemoriesResponse,
    ChatRequest,
    ChatResponse,
    ClearMemoriesResponse,
    GetMemoriesResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Chat & Memory"])


# ── POST /api/chat ───────────────────────────────────────────

@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Answer a question from stored memories (RAG)",
    description=(
        "Asks a question against the in-memory vector store using "
        "semantic search (all-MiniLM-L6-v2) + NVIDIA Nemotron answer generation. "
        "Schedule queries (e.g. 'What do I have tomorrow?') are handled via "
        "structured search across all modes."
    ),
)
async def chat(body: ChatRequest) -> ChatResponse:
    from services.chat_service import answer_question  # type: ignore

    try:
        answer = answer_question(
            question=body.question,
            mode=body.mode,
            user_id=body.user_id,
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        )
    except Exception as exc:
        logger.exception("Chat error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat failed: {type(exc).__name__}",
        )

    return ChatResponse(answer=answer, mode=body.mode)


# ── POST /api/memories ───────────────────────────────────────

@router.post(
    "/memories",
    response_model=AddMemoriesResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add memories to the in-memory store",
    description=(
        "Store normalized memory objects extracted from a conversation. "
        "Each memory must include at least `text` and `memory_type`. "
        "The `mode` field tags every memory with the current application mode."
    ),
)
async def add_memories(body: AddMemoriesRequest) -> AddMemoriesResponse:
    from services.chat_service import add_memories as _add  # type: ignore

    raw = [m.model_dump() for m in body.memories]

    try:
        added = _add(raw, mode=body.mode, user_id=body.user_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        logger.exception("Add memories error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add memories: {type(exc).__name__}",
        )

    return AddMemoriesResponse(added=added, mode=body.mode)


# ── GET /api/memories ────────────────────────────────────────

@router.get(
    "/memories",
    response_model=GetMemoriesResponse,
    summary="Get stored memories",
    description="Return all stored memories, optionally filtered by `mode` and `user_id`.",
)
async def get_memories(
    mode: Optional[str] = Query(
        default=None,
        description="Filter by mode: general, hospital, meeting, classroom"
    ),
    user_id: Optional[str] = Query(
        default=None,
        description="Filter by user ID"
    ),
) -> GetMemoriesResponse:
    from services.chat_service import get_memories as _get  # type: ignore

    try:
        memories = _get(mode=mode, user_id=user_id)
    except Exception as exc:
        logger.exception("Get memories error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get memories: {type(exc).__name__}",
        )

    return GetMemoriesResponse(memories=memories, count=len(memories), mode=mode)


# ── DELETE /api/memories ─────────────────────────────────────

@router.delete(
    "/memories",
    response_model=ClearMemoriesResponse,
    summary="Clear stored memories",
    description="Delete all stored memories for the given user (or all if no user_id).",
)
async def clear_memories(
    user_id: Optional[str] = Query(
        default=None,
        description="Clear only this user's memories; omit to clear all."
    ),
) -> ClearMemoriesResponse:
    from services.chat_service import clear_memories as _clear  # type: ignore

    try:
        _clear(user_id=user_id)
    except Exception as exc:
        logger.exception("Clear memories error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear memories: {type(exc).__name__}",
        )

    return ClearMemoriesResponse(cleared=True, user_id=user_id)
