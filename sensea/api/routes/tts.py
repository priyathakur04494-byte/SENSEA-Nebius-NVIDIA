
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from services import tts_service


router = APIRouter(prefix="/api/tts", tags=["Text to Speech"])


class TTSRequest(BaseModel):
    text: str


@router.post("/")
async def generate_speech_endpoint(request: TTSRequest):
    try:
        audio_bytes = tts_service.generate_speech(request.text)

        return Response(
            content=audio_bytes,
            media_type="audio/wav",
            headers={
                "Content-Disposition": 'attachment; filename="sensea_tts.wav"'
            }
        )

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
