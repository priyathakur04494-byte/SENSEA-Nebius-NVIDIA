from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel


class UserCreate(BaseModel):
    email: Optional[str] = None
    display_name: Optional[str] = None


class UserUpdate(BaseModel):
    email: Optional[str] = None
    display_name: Optional[str] = None


class UserResponse(BaseModel):
    id: UUID
    email: Optional[str] = None
    display_name: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True