"""Phase 4 backend — FastAPI app.

Lifespan creates the table and loads the Phase 1 checkpoint; the routes live in
``backend/routes/``. See docs/backend-roadmap.md.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .database import init_schema
from .routes import analyze, history
from .services import detector

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_schema()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    detector.load_model(settings.MODEL_DIR)
    yield


app = FastAPI(title="Deepfake Detection & Alert System", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze.router)
app.include_router(history.router)


@app.get("/health", tags=["meta"])
def health() -> dict:
    """Liveness plus whether a real checkpoint is loaded (vs. stub mode)."""
    return {"status": "ok", "model_loaded": detector.MODEL is not None}
