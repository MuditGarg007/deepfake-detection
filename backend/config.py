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
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
