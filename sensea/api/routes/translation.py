from fastapi import APIRouter, HTTPException

from schemas.translation import TranslationRequest
from services import translation_service

router = APIRouter(prefix="/api", tags=["Translation"])


@router.post("/translation")
def translate_endpoint(request: TranslationRequest):
    try:
        translated_text = translation_service.translate_text(
            text=request.text,
            source_language=request.source_language,
            target_language=request.target_language
        )

        return {
            "translated_text": translated_text,
            "source_language": request.source_language,
            "target_language": request.target_language
        }

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))