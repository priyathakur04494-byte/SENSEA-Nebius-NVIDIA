
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from services import speech_service

router = APIRouter(prefix="/api/speech", tags=["Speech"])


@router.post("/transcribe")
async def transcribe_endpoint(
    audio: UploadFile = File(...),
    language: str = Form("eng_Latn"),
):
    try:
        audio_bytes = await audio.read()

        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Uploaded audio is empty.")

        return speech_service.transcribe_audio(
            audio_bytes=audio_bytes,
            filename=audio.filename or "recording.wav",
            language=language,
        )

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    except RuntimeError as error:
        raise HTTPException(status_code=500, detail=str(error))

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
