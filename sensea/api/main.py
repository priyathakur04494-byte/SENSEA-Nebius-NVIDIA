from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.conversation import router as conversation_router
from routes.translation import router as translation_router
import config
from routes.chat import router as chat_router
from routes.speech import router as speech_router
from routes.tts import router as tts_router

app = FastAPI(
    title="SENSEA API",
    description="SENSEA ML Backend API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)
app.include_router(conversation_router)
app.include_router(translation_router)
app.include_router(speech_router)
app.include_router(tts_router)


@app.get("/")
def root():
    return {
        "message": "SENSEA API is running"
    }


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "service": "SENSEA API"
    }