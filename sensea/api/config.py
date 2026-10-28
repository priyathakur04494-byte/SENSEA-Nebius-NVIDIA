# ============================================================
# SENSEA API — Configuration
# ============================================================
#
# All settings come from environment variables.
# Copy .env.example to .env and fill in the values.
# ============================================================

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# ── Locate .env ─────────────────────────────────────────────
# Support three layouts:
#   1.  sensea/api/.env            (api-local)
#   2.  sensea/.env                (monorepo root)
#   3.  repo root .env             (top-level)

_HERE = Path(__file__).resolve().parent          # sensea/api/
_SENSEA_DIR = _HERE.parent                       # sensea/
_REPO_DIR = _SENSEA_DIR.parent                   # repo root

for _candidate in [
    _HERE / ".env",
    _SENSEA_DIR / ".env",
    _REPO_DIR / ".env",
]:
    if _candidate.is_file():
        load_dotenv(_candidate)
        break
else:
    # Last resort: let python-dotenv search upward
    load_dotenv()


# ── ML Backend path on sys.path ─────────────────────────────
ML_BACKEND_DIR = Path(os.getenv("ML_BACKEND_DIR", str(_SENSEA_DIR / "ML Backend")))

if str(ML_BACKEND_DIR) not in sys.path:
    sys.path.append(str(ML_BACKEND_DIR))

# Sign-language sub-package also needs to be importable
_SIGN_LANG_DIR = ML_BACKEND_DIR / "sign_language"
if str(_SIGN_LANG_DIR) not in sys.path:
    sys.path.append(str(_SIGN_LANG_DIR))


# ── NVIDIA / Nemotron ───────────────────────────────────────
NVIDIA_API_KEY: str = os.getenv("NVIDIA_API_KEY", "")

# ── API Access Security ─────────────────────────────────────
# Empty string = no auth required (dev-friendly default).
# Set a strong random value in production.
SENSEA_API_KEY: str = os.getenv("SENSEA_API_KEY", "")

# ── Whisper.cpp ─────────────────────────────────────────────
# Path to the compiled whisper-cli binary.
# On macOS: ~/whisper.cpp/build/bin/whisper-cli
# On Windows: C:/whisper.cpp/build/bin/Release/whisper-cli.exe
WHISPER_CPP_PATH: str = os.getenv(
    "WHISPER_CPP_PATH",
    str(Path.home() / "whisper.cpp"),
)

# ── Whisper models ──────────────────────────────────────────
WHISPER_SMALL_MODEL: str = os.getenv("WHISPER_SMALL_MODEL", "small")
WHISPER_MEDIUM_MODEL: str = os.getenv("WHISPER_MEDIUM_MODEL", "medium")

# ── Sign-language model ─────────────────────────────────────
SIGN_LANGUAGE_MODEL_PATH: str = os.getenv(
    "SIGN_LANGUAGE_MODEL_PATH",
    str(ML_BACKEND_DIR / "sign_language" / "models" / "sign_language_model.pth"),
)

# ── MediaPipe task files ─────────────────────────────────────
MEDIAPIPE_DIR: str = os.getenv(
    "MEDIAPIPE_DIR",
    str(ML_BACKEND_DIR / "sign_language" / "models" / "mediapipe"),
)

# ── Server ──────────────────────────────────────────────────
HOST: str = os.getenv("SENSEA_HOST", "0.0.0.0")
PORT: int = int(os.getenv("SENSEA_PORT", "8000"))

# ── CORS ────────────────────────────────────────────────────
# Comma-separated list of allowed origins.
# "*" allows all (fine for local dev; restrict in prod).
CORS_ORIGINS_RAW: str = os.getenv("CORS_ORIGINS", "*")

if CORS_ORIGINS_RAW.strip() == "*":
    CORS_ORIGINS: list[str] = ["*"]
else:
    CORS_ORIGINS = [
        o.strip()
        for o in CORS_ORIGINS_RAW.split(",")
        if o.strip()
    ]

# ── Logging ─────────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

# ── Sanity check ────────────────────────────────────────────

def check_required() -> list[str]:
    """Return a list of missing required environment variables."""
    missing = []
    if not NVIDIA_API_KEY:
        missing.append("NVIDIA_API_KEY")
    # Warn if API key is not set in production
    sensea_env = os.getenv("SENSEA_ENV", "development")
    if not SENSEA_API_KEY and sensea_env != "development":
        missing.append(
            "SENSEA_API_KEY (strongly recommended in production; "
            "currently empty — all endpoints are publicly accessible)"
        )
    return missing


def validate_assets() -> list[str]:
    """
    Check that Whisper binary and model files exist.

    Returns a list of warning strings (not errors); startup still
    proceeds so the server can serve non-speech endpoints.
    """
    warnings: list[str] = []

    base = Path(WHISPER_CPP_PATH)

    # whisper-cli binary (try Windows paths first, then Unix)
    win_paths = [
        base / "build" / "bin" / "Release" / "whisper-cli.exe",
        base / "build" / "bin" / "whisper-cli.exe",
    ]
    unix_path = base / "build" / "bin" / "whisper-cli"

    binary_found = any(p.exists() for p in win_paths) or unix_path.exists()
    if not binary_found:
        warnings.append(
            f"whisper-cli binary not found under {WHISPER_CPP_PATH}. "
            "Speech transcription will be unavailable."
        )

    # ggml model files
    for model_name in (WHISPER_SMALL_MODEL, WHISPER_MEDIUM_MODEL):
        model_path = base / "models" / f"ggml-{model_name}.bin"
        if not model_path.exists():
            warnings.append(
                f"Whisper model file not found: {model_path}. "
                f"Transcription with the '{model_name}' model will fail."
            )

    return warnings
