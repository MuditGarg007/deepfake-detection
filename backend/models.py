"""SQLAlchemy models."""

from datetime import datetime
from pathlib import Path

from sqlalchemy import JSON, TIMESTAMP, CheckConstraint, Float, Integer, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .database import Base


class Analysis(Base):
    __tablename__ = "analyses"
    __table_args__ = (
        CheckConstraint(
            "status IN ('REAL','SUSPICIOUS','HIGH_RISK')", name="ck_analyses_status"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    filename: Mapped[str] = mapped_column(Text, nullable=False)
    # Where the uploaded file was kept so it can be played back later. Nullable:
    # rows written before playback existed have no path, and a file can be
    # cleaned off disk without invalidating the analysis.
    storage_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    fake_probability: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    suspicious_start: Mapped[float | None] = mapped_column(Float, nullable=True)
    suspicious_end: Mapped[float | None] = mapped_column(Float, nullable=True)
    # JSONB on Postgres (the roadmap's schema), plain JSON on the SQLite fallback.
    frame_scores: Mapped[list] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"), nullable=False, default=list
    )
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )

    @property
    def has_video(self) -> bool:
        """Whether the source video is still on disk and can be streamed back."""
        return self.storage_path is not None and Path(self.storage_path).is_file()
