from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.conversations import router as conversations_router
from app.api.users import router as users_router
from app.api.captions import router as captions_router
from app.api.speakers import router as speakers_router

app = FastAPI(
    title="Signal Accessibility Assistant API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:19000", "http://10.0.2.2:19000", "http://localhost:19000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(users_router)
app.include_router(conversations_router)
app.include_router(captions_router)
app.include_router(speakers_router)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "signal-backend",
    }


