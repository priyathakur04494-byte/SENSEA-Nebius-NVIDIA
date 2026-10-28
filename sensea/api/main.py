# ============================================================
# SENSEA — FastAPI Application Entry Point
# ============================================================
#
# Run locally:
#   cd sensea/api
#   uvicorn main:app --reload --host 0.0.0.0 --port 8000
#
# Or from the repo root:
#   uvicorn sensea.api.main:app --reload
#
# From Expo (development):
#   Replace localhost with your machine's local IP address.
#   e.g. http://192.168.1.42:8000
# ============================================================

from __future__ import annotations

import hmac
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# ── Bootstrap: ensure ML Backend + sign_language are importable ─
# config.py inserts paths into sys.path at import time.
_API_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_API_DIR))

import config  # noqa: E402  (must be before ML imports)

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ── Logging ─────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("sensea.api")


# ── Startup / shutdown ───────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load heavy ML models at startup so every request is fast.

    Each service module handles its own lazy initialisation;
    we call init functions here to front-load the cost.
    """
    logger.info("=== SENSEA API starting ===")

    missing = config.check_required()
    if missing:
        logger.warning(
            "Missing required environment variables: %s. "
            "Some features may be unavailable.",
            ", ".join(missing),
        )

    # ── Asset validation ─────────────────────────────────────
    for warning in config.validate_assets():
        logger.warning("Asset check: %s", warning)

    # ── Memory / RAG embeddings ──────────────────────────────
    try:
        from services.chat_service import _init_embeddings  # type: ignore
        _init_embeddings()
    except Exception as exc:
        logger.warning("Memory store init skipped: %s", exc)

    # ── Speaker diarization ──────────────────────────────────
    try:
        from services.speech_service import _init_diarization  # type: ignore
        _init_diarization()
    except Exception as exc:
        logger.warning("Diarization init skipped: %s", exc)

    # ── Sign language model ──────────────────────────────────
    try:
        from services.sign_language_service import _init_sign_language  # type: ignore
        _init_sign_language()
    except Exception as exc:
        logger.warning("Sign language model init skipped: %s", exc)

    logger.info("=== SENSEA API ready ===")
    yield

    logger.info("=== SENSEA API shutting down ===")


# ── Application ──────────────────────────────────────────────

app = FastAPI(
    title="SENSEA ML Backend API",
    description=(
        "Production REST API for the SENSEA assistive-technology platform.\n\n"
        "Exposes speech-to-text (Whisper), translation (IndicTrans2), "
        "text-to-speech (Kokoro), speaker diarization (ECAPA-TDNN), "
        "conversation analysis (NVIDIA Nemotron), RAG-based chat, "
        "and sign language recognition (MediaPipe + Conv1D+BiGRU).\n\n"
        "**React Native Expo client**: set `BASE_URL` to your machine's LAN IP, "
        "e.g. `http://192.168.1.42:8000`."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ── CORS ─────────────────────────────────────────────────────
# Allow all origins in development; restrict via CORS_ORIGINS env var in prod.

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)


# ── API-key security middleware ───────────────────────────────
# If SENSEA_API_KEY is empty, all requests pass through (dev-friendly).
# Otherwise every request (except public paths) must supply the correct
# X-API-Key header; mismatches receive a constant-time 401.

_PUBLIC_PATHS = {"/", "/docs", "/redoc", "/openapi.json", "/api/health"}


@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    api_key = config.SENSEA_API_KEY
    # Skip auth for:
    #  - public/docs paths
    #  - OPTIONS preflight (CORS) — these never carry X-API-Key and must pass through
    #    so that browsers can complete the CORS handshake before the real request
    if (
        not api_key
        or request.method == "OPTIONS"
        or request.url.path in _PUBLIC_PATHS
    ):
        return await call_next(request)

    client_key = request.headers.get("X-API-Key", "")
    if not hmac.compare_digest(
        client_key.encode("utf-8"),
        api_key.encode("utf-8"),
    ):
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"detail": "Unauthorized"},
        )
    return await call_next(request)


# ── Global exception handler ─────────────────────────────────

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(
        "Unhandled exception on %s %s",
        request.method,
        request.url.path,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": f"Internal server error: {type(exc).__name__}"},
    )


# ── Routers ──────────────────────────────────────────────────

from routes.health import router as health_router  # noqa: E402
from routes.speech import router as speech_router  # noqa: E402
from routes.translation import router as translation_router  # noqa: E402
from routes.conversation import router as conversation_router  # noqa: E402
from routes.chat import router as chat_router  # noqa: E402
from routes.sign_language import router as sign_language_router  # noqa: E402
from routes.pipeline import router as pipeline_router  # noqa: E402

app.include_router(health_router)
app.include_router(speech_router)
app.include_router(translation_router)
app.include_router(conversation_router)
app.include_router(chat_router)
app.include_router(sign_language_router)
app.include_router(pipeline_router)


# ── Root redirect ─────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root() -> JSONResponse:
    return JSONResponse(
        {"message": "SENSEA API is running. See /docs for the API reference."}
    )


# ── Dev entrypoint ────────────────────────────────────────────
# For development: python main.py  — set SENSEA_ENV=development for reload
#
# For prod, run:
#   uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1 --no-access-log

if __name__ == "__main__":
    import uvicorn  # type: ignore

    _dev = os.getenv("SENSEA_ENV", "development") == "development"

    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        reload=_dev,
        log_level=config.LOG_LEVEL.lower(),
    )
