from typing import Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str
    mode: str = "general"


class ChatResponse(BaseModel):
    answer: str
    mode: str


class MemoryItem(BaseModel):
    text: str
    memory_type: str = "general"
    importance: Optional[str] = "medium"
    date: Optional[str] = None
    time: Optional[str] = None


class AddMemoriesRequest(BaseModel):
    memories: list[MemoryItem]
    mode: str = "general"


class AddMemoriesResponse(BaseModel):
    added: int
    mode: str


class GetMemoriesResponse(BaseModel):
    memories: list[dict]
    count: int
    mode: Optional[str] = None


class ClearMemoriesResponse(BaseModel):
    cleared: int