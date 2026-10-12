import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from . import config
from .auth import RequireUser, frontend_api
from .database import init_schema
from .routes import admin, analyze, history, live, users
from .services import detector

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_schema()
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    detector.load_model(config.MODEL_DIR)
    yield


app = FastAPI(title="Deepfake Detection & Alert System", lifespan=lifespan)

app.add_middleware(RequireUser)

app.include_router(admin.router, prefix="/api")
app.include_router(users.router, prefix="/api")
app.include_router(analyze.router, prefix="/api")
app.include_router(history.router, prefix="/api")
app.include_router(live.router, prefix="/api")


@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok", "model_loaded": detector.MODEL is not None}


@app.get("/api/config", tags=["meta"])
def client_config():
    key = config.CLERK_PUBLISHABLE_KEY
    return {"publishable_key": key, "frontend_api": frontend_api(key) if key else ""}


app.mount("/", StaticFiles(directory=config.BACKEND_DIR / "static", html=True), name="static")
