"""Phase 4 backend — FastAPI app.

T1: minimal app that boots and serves GET /health.
T3: lifespan creates tables on startup.
T7 will add the real routes + CORS.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from . import models  # noqa: F401  (register tables on Base)
from .config import settings
from .database import Base, engine
from .services import detector


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    detector.load_model(settings.MODEL_DIR)
    yield


app = FastAPI(title="Deepfake Detection & Alert System", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}