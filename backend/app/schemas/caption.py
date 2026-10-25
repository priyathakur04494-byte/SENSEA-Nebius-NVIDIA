from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel


class CaptionCreate(BaseModel):
    conversation_id: UUID
    speaker_id: Optional[UUID] = None
    original_text: str
    simplified_text: Optional[str] = None
    translated_text: Optional[str] = None
    language: str = "en"
    sequence_number: int
    is_final: bool = False


class CaptionResponse(BaseModel):
    id: UUID
    conversation_id: UUID
    speaker_id: Optional[UUID] = None
    original_text: str
    simplified_text: Optional[str] = None
    translated_text: Optional[str] = None
    language: str
    sequence_number: int
    is_final: bool
    timestamp: datetime

    class Config:
        from_attributes = True


class CaptionUpdate(BaseModel):
    original_text: Optional[str] = None
    simplified_text: Optional[str] = None
    translated_text: Optional[str] = None
    language: Optional[str] = None
    sequence_number: Optional[int] = None
    is_final: Optional[bool] = None