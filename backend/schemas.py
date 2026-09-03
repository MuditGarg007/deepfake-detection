"""Pydantic response schemas (T4)."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class FrameScore(BaseModel):
    timestamp: float
    fake_probability: float


class AnalysisOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    fake_probability: float
    status: str
    suspicious_start: float | None
    suspicious_end: float | None
    frame_scores: list[FrameScore]
    created_at: datetime
    # Drives the player in the UI — false once the file is gone from disk.
    has_video: bool


class HistoryOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    fake_probability: float
    status: str
    created_at: datetime
    has_video: bool
