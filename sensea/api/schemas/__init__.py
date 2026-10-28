# ============================================================
# SENSEA API — Pydantic Schemas
# ============================================================
# Routes import everything from here:  `from schemas import ...`
# ============================================================

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# ── Health ───────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "SENSEA ML Backend"
    version: str = "1.0.0"
    whisper_cli_ok: Optional[bool] = None
    sign_language_model_ok: Optional[bool] = None
    nvidia_key_set: Optional[bool] = None
    api_key_required: Optional[bool] = None


# ── Speech: transcribe ───────────────────────────────────────

class TranscribeResponse(BaseModel):
    transcript: str
    language: str = Field(..., description="Human-readable name, e.g. 'English'")
    whisper_lang: str = Field(..., description="Whisper code, e.g. 'en', 'hi'")


# ── Speech: diarize ──────────────────────────────────────────

class SpeakerSegment(BaseModel):
    speaker_id: str
    is_new: bool
    confidence: float
    duration_seconds: float


class DiarizeResponse(BaseModel):
    speakers: List[SpeakerSegment]
    total_speakers: int


# ── Translation ──────────────────────────────────────────────

class TranslationRequest(BaseModel):
    text: str = Field(..., min_length=1)
    source_language: str = Field(..., examples=["eng_Latn"])
    target_language: str = Field(..., examples=["hin_Deva"])


class TranslationResponse(BaseModel):
    translation: str
    source_language: str
    target_language: str


# ── Languages ────────────────────────────────────────────────

class LanguageInfo(BaseModel):
    code: str
    name: str


class LanguagesResponse(BaseModel):
    speech_languages: List[LanguageInfo]
    translation_targets: List[LanguageInfo]


# ── Conversation analysis ────────────────────────────────────

class ConversationSegment(BaseModel):
    speaker_id: str = "Speaker"
    text: str
    translation: str = ""


class AnalyzeRequest(BaseModel):
    mode: str = Field(..., examples=["hospital"],
                      description="general | hospital | meeting | classroom")
    segments: List[ConversationSegment] = Field(..., min_length=1)


class AnalyzeResponse(BaseModel):
    mode: str
    result: Dict[str, Any]


# ── Chat / memories ──────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    mode: str = "general"
    user_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    mode: str


class MemoryItem(BaseModel):
    # Extractor JSON may carry extra keys; keep them for the normalizer.
    model_config = ConfigDict(extra="allow")

    text: str = Field(..., min_length=1)
    memory_type: str
    importance: Optional[str] = None
    date: Optional[str] = None
    time: Optional[str] = None


class AddMemoriesRequest(BaseModel):
    memories: List[MemoryItem] = Field(..., min_length=1)
    mode: str = "general"
    user_id: Optional[str] = None


class AddMemoriesResponse(BaseModel):
    added: int
    mode: str


class GetMemoriesResponse(BaseModel):
    memories: List[Dict[str, Any]]
    count: int
    mode: Optional[str] = None


class ClearMemoriesResponse(BaseModel):
    cleared: bool
    user_id: Optional[str] = None


# ── Sign language ────────────────────────────────────────────

class SignPrediction(BaseModel):
    label: str
    confidence: float
    all_scores: Dict[str, float]


class SignPredictResponse(BaseModel):
    prediction: SignPrediction
    frames_used: int
    orientation: str


# ── Pipeline ─────────────────────────────────────────────────

class PipelineSegment(BaseModel):
    speaker_id: str
    text: str
    translation: str = ""
    confidence: float = 0.0
    duration_seconds: float = 0.0


class PipelineResponse(BaseModel):
    segments: List[PipelineSegment]
    total_speakers: int
