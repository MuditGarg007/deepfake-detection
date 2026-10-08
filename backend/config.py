import os
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent

load_dotenv(BACKEND_DIR / ".env")

DATABASE_URL = os.getenv("NEON_DB_URL") or os.getenv("DATABASE_URL") or ""
DATABASE_URL = DATABASE_URL.replace("+psycopg2", "")

MODEL_DIR = os.getenv("MODEL_DIR", "machine-learning/checkpoints")
UPLOAD_DIR = BACKEND_DIR / os.getenv("UPLOAD_DIR", "uploads")
RISK_SUSPICIOUS = float(os.getenv("RISK_SUSPICIOUS", "0.4"))
RISK_HIGH = float(os.getenv("RISK_HIGH", "0.7"))
FRAME_THRESHOLD = float(os.getenv("FRAME_THRESHOLD", "0.7"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "200"))

RISK_HYSTERESIS = float(os.getenv("RISK_HYSTERESIS", "0.1"))
LIVE_EMA_ALPHA = float(os.getenv("LIVE_EMA_ALPHA", "0.2"))
LIVE_HISTORY_FRAMES = int(os.getenv("LIVE_HISTORY_FRAMES", "120"))
LIVE_DETECT_EVERY = int(os.getenv("LIVE_DETECT_EVERY", "5"))
LIVE_SESSION_TTL = float(os.getenv("LIVE_SESSION_TTL", "300"))
LIVE_TIMELINE_MAX = int(os.getenv("LIVE_TIMELINE_MAX", "4000"))
CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000,http://localhost:8501,http://127.0.0.1:8501",
).split(",")
