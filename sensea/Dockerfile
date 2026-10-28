# ============================================================
# SENSEA ML Backend — Docker Image
# ============================================================
#
# Build:
#   docker build -t sensea-api -f sensea/Dockerfile .
#
# Run:
#   docker run -p 8000:8000 \
#     -e NVIDIA_API_KEY=nvapi-... \
#     -e SENSEA_API_KEY=your-secret \
#     -e CORS_ORIGINS=http://192.168.1.42:8081 \
#     -v /host/whisper.cpp:/opt/whisper.cpp \
#     -v /host/models:/opt/models \
#     sensea-api
#
# NOTE: whisper-cli must be pre-built and mounted at /opt/whisper.cpp,
#       or set WHISPER_CPP_PATH to point to the binary location.
# ============================================================

FROM python:3.11-slim

# ── System deps ─────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    libsndfile1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Copy ML Backend source ───────────────────────────────────
COPY sensea/ /app/sensea/

# ── Install Python deps ──────────────────────────────────────
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r /app/sensea/api/requirements.txt \
 && pip install --no-cache-dir \
        git+https://github.com/VarunGumma/IndicTransToolkit.git

# ── Pre-download HuggingFace models (baked into image) ───────
# Uncomment and set HF_TOKEN if models are gated.
# ENV HF_TOKEN=hf_...
# RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# ── Environment defaults ─────────────────────────────────────
ENV SENSEA_HOST=0.0.0.0
ENV SENSEA_PORT=8000
ENV SENSEA_ENV=production
ENV ML_BACKEND_DIR=/app/sensea/ML\ Backend
ENV WHISPER_CPP_PATH=/opt/whisper.cpp
ENV SIGN_LANGUAGE_MODEL_PATH=/opt/models/sign_language_model.pth
ENV MEDIAPIPE_DIR=/opt/models/mediapipe
ENV LOG_LEVEL=INFO

EXPOSE 8000

# ── Production startup ───────────────────────────────────────
# Single worker — in-memory state (RAG, diarization) is not shared
# across workers. Use external storage (Redis/pgvector) before scaling.
CMD ["uvicorn", "sensea.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--no-access-log"]
