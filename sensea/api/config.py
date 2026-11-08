import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Project directories
API_DIR = Path(__file__).resolve().parent
SENSEA_DIR = API_DIR.parent
REPO_DIR = SENSEA_DIR.parent

# Load environment variables
for env_file in [
    API_DIR / ".env",
    SENSEA_DIR / ".env",
    REPO_DIR / ".env",
]:
    if env_file.is_file():
        load_dotenv(env_file)
        break
else:
    load_dotenv()

# Simran's ML Backend directory
ML_BACKEND_DIR = SENSEA_DIR / "ML Backend"

# Make ML modules importable
if str(ML_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(ML_BACKEND_DIR))

# NVIDIA API key
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")

# API server settings
HOST = os.getenv("SENSEA_HOST", "0.0.0.0")
PORT = int(os.getenv("SENSEA_PORT", "8000"))

# CORS settings
CORS_ORIGINS_RAW = os.getenv("CORS_ORIGINS", "*")

if CORS_ORIGINS_RAW.strip() == "*":
    CORS_ORIGINS = ["*"]
else:
    CORS_ORIGINS = [
        origin.strip()
        for origin in CORS_ORIGINS_RAW.split(",")
        if origin.strip()
    ]