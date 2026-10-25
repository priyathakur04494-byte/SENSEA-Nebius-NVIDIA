from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel


class ConversationCreate(BaseModel):
    user_id: UUID
    title: Optional[str] = None


class ConversationResponse(BaseModel):
    id: UUID
    user_id: UUID
    title: Optional[str] = None
    started_at: datetime
    ended_at: Optional[datetime] = None
    created_at: datetime

    class Config:
        from_attributes = True


class ConversationUpdate(BaseModel):
    title: Optional[str] = None