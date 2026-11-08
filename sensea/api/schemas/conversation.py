from typing import Literal, Optional
from pydantic import BaseModel


class ConversationSegment(BaseModel):
    speaker_id: str = "Unknown Speaker"
    text: str
    translation: Optional[str] = ""


class ConversationAnalysisRequest(BaseModel):
    mode: Literal["general", "hospital", "meeting", "classroom"]
    segments: list[ConversationSegment]


class ConversationAnalysisResponse(BaseModel):
    mode: str
    analysis: dict