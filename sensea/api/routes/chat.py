from fastapi import APIRouter, HTTPException
from typing import Optional

from schemas.chat import (
    ChatRequest,
    ChatResponse,
    AddMemoriesRequest,
    AddMemoriesResponse,
    GetMemoriesResponse,
    ClearMemoriesResponse,
)

from services import chat_service

router = APIRouter(prefix="/api", tags=["Chat & Memory"])


@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    try:
        answer = chat_service.answer_question(
            request.question,
            request.mode
        )
        return ChatResponse(answer=answer, mode=request.mode)

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))


@router.post("/memories", response_model=AddMemoriesResponse)
def add_memories_endpoint(request: AddMemoriesRequest):
    try:
        added = chat_service.add_memories(
            [memory.model_dump() for memory in request.memories],
            request.mode
        )
        return AddMemoriesResponse(added=added, mode=request.mode)

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))


@router.get("/memories", response_model=GetMemoriesResponse)
def get_memories_endpoint(mode: Optional[str] = None):
    try:
        memories = chat_service.get_memories(mode)
        return GetMemoriesResponse(
            memories=memories,
            count=len(memories),
            mode=mode
        )

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))


@router.delete("/memories", response_model=ClearMemoriesResponse)
def clear_memories_endpoint():
    cleared = chat_service.clear_memories()
    return ClearMemoriesResponse(cleared=cleared)