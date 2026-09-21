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

# Live screen-share tuning. The entry thresholds above are shared with the
# upload path so the two verdicts mean the same thing; only the exits are
# sticky, by RISK_HYSTERESIS, so a score parked on a threshold cannot make the
# badge strobe between two states several times a second.
RISK_HYSTERESIS = float(os.getenv("RISK_HYSTERESIS", "0.1"))
# Alpha for the exponential moving average the badge reads. Raw per-frame
# scores flicker hard - one blurred frame swings the number - so nothing shows
# them directly.
LIVE_EMA_ALPHA = float(os.getenv("LIVE_EMA_ALPHA", "0.2"))
# Sparkline window: 120 frames at 2 fps is the last 60 seconds.
LIVE_HISTORY_FRAMES = int(os.getenv("LIVE_HISTORY_FRAMES", "120"))
# MTCNN runs on every Nth frame and the box is reused in between; detection
# dominates the per-frame cost and a face does not teleport in 500 ms.
LIVE_DETECT_EVERY = int(os.getenv("LIVE_DETECT_EVERY", "5"))
# An abandoned tab stops posting frames without ever calling /stop.
LIVE_SESSION_TTL = float(os.getenv("LIVE_SESSION_TTL", "300"))
# Cap on the stored timeline. Past it the timeline is halved in resolution
# rather than truncated, so a long session keeps its whole shape.
LIVE_TIMELINE_MAX = int(os.getenv("LIVE_TIMELINE_MAX", "4000"))
# The live screen-share component posts straight from the browser, so the
# Streamlit origin needs to be allowed alongside the old Next.js one.
CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000,http://localhost:8501,http://127.0.0.1:8501",
).split(",")
