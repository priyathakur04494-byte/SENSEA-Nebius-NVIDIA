import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SENSEA_DIR = _ROOT / "sensea"
_API_DIR = _SENSEA_DIR / "api"

for _p in (_ROOT, _SENSEA_DIR, _API_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from sensea.main import app  # noqa: F401

if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.getenv("PORT", os.getenv("SENSEA_PORT", "8000")))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
