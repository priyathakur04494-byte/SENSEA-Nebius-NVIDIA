# ============================================================
# SENSEA API — Health Route
# ============================================================

import sys
from pathlib import Path

from fastapi import APIRouter
from schemas import HealthResponse

router = APIRouter(prefix="/api", tags=["Health"])


@router.get("/health", response_model=HealthResponse, summary="Health check")
async def health() -> HealthResponse:
    """Return service health status including asset and key availability."""
    # Import config here (late import keeps this module side-effect-free at load)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import config  # type: ignore

    base = Path(config.WHISPER_CPP_PATH)

    # whisper-cli binary (Windows + Unix paths)
    whisper_cli_ok = any([
        (base / "build" / "bin" / "Release" / "whisper-cli.exe").exists(),
        (base / "build" / "bin" / "whisper-cli.exe").exists(),
        (base / "build" / "bin" / "whisper-cli").exists(),
    ])

    sign_model_path = Path(config.SIGN_LANGUAGE_MODEL_PATH)
    sign_language_model_ok = sign_model_path.exists()

    return HealthResponse(
        status="ok",
        service="SENSEA ML Backend",
        version="1.0.0",
        whisper_cli_ok=whisper_cli_ok,
        sign_language_model_ok=sign_language_model_ok,
        nvidia_key_set=bool(config.NVIDIA_API_KEY),
        api_key_required=bool(config.SENSEA_API_KEY),
    )
