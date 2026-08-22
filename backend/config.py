"""Pydantic-settings config read from backend/.env."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_BACKEND_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    NEON_DB_URL: str
    MODEL_DIR: str = "machine-learning/checkpoints"  # relative to project root
    UPLOAD_DIR: str = "uploads"
    RISK_SUSPICIOUS: float = 0.4
    RISK_HIGH: float = 0.7
    FRAME_THRESHOLD: float = 0.7
    MAX_UPLOAD_MB: int = 200


settings = Settings()