from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel


class SpeakerCreate(BaseModel):
    user_id: UUID
    name: Optional[str] = None
    label: str
    color_tag: Optional[str] = None
    is_important_person: bool = False


class SpeakerUpdate(BaseModel):
    name: Optional[str] = None
    label: Optional[str] = None
    color_tag: Optional[str] = None
    is_important_person: Optional[bool] = None


class SpeakerResponse(BaseModel):
    id: UUID
    user_id: UUID
    name: Optional[str] = None
    label: str
    color_tag: Optional[str] = None
    is_important_person: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True