from pydantic import BaseModel


class TranslationRequest(BaseModel):
    text: str
    source_language: str = "en"
    target_language: str