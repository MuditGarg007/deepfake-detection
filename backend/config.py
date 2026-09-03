"""Pydantic-settings config read from backend/.env."""

from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_BACKEND_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # The Neon connection string. `DATABASE_URL` is accepted as an alias since
    # that is the name used in docs/backend-roadmap.md §3.4.
    NEON_DB_URL: str | None = Field(
        default=None,
        validation_alias=AliasChoices("NEON_DB_URL", "DATABASE_URL"),
    )
    MODEL_DIR: str = "machine-learning/checkpoints"  # relative to project root
    UPLOAD_DIR: str = "uploads"
    RISK_SUSPICIOUS: float = 0.4
    RISK_HIGH: float = 0.7
    FRAME_THRESHOLD: float = 0.7
    MAX_UPLOAD_MB: int = 200
    # Phase 5 frontend origins allowed by CORS.
    CORS_ORIGINS: str = "http://localhost:3000,http://127.0.0.1:3000"

    @property
    def database_url(self) -> str:
        """Neon when configured, otherwise a local SQLite file.

        The fallback keeps the server (and the smoke test) runnable on a fresh
        clone with no credentials; production/demo runs set NEON_DB_URL.
        """
        if self.NEON_DB_URL:
            return self.NEON_DB_URL
        return f"sqlite:///{_BACKEND_DIR / 'app.db'}"

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]

    @property
    def upload_dir(self) -> Path:
        """UPLOAD_DIR, resolved relative to backend/ when it is not absolute."""
        directory = Path(self.UPLOAD_DIR)
        if not directory.is_absolute():
            directory = _BACKEND_DIR / directory
        return directory


settings = Settings()
