from pydantic import BaseModel
from typing import Optional


class InputData(BaseModel):

    user_id: str

    session_id: str

    mode: str

    source_language: str

    timestamp: str

    transcript: str

    translation: Optional[str] = None