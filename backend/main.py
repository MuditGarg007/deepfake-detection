import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import config
from .database import init_schema
from .routes import analyze, history
from .services import detector

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_schema()
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    detector.load_model(config.MODEL_DIR)
    yield


app = FastAPI(title="Deepfake Detection & Alert System", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze.router)
app.include_router(history.router)


@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok", "model_loaded": detector.MODEL is not None}
