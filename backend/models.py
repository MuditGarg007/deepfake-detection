"""SQLAlchemy models."""

from datetime import datetime

from sqlalchemy import JSON, TIMESTAMP, CheckConstraint, Float, Integer, Text, func
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
    fake_probability: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    suspicious_start: Mapped[float | None] = mapped_column(Float, nullable=True)
    suspicious_end: Mapped[float | None] = mapped_column(Float, nullable=True)
    frame_scores: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )