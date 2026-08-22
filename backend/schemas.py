"""Pydantic response schemas (T4)."""

from datetime import datetime

from pydantic import BaseModel


class FrameScore(BaseModel):
    timestamp: float
    fake_probability: float


class AnalysisOut(BaseModel):
    id: int
    filename: str
    fake_probability: float
    status: str
    suspicious_start: float | None
    suspicious_end: float | None
    frame_scores: list[FrameScore]
    created_at: datetime


class HistoryOut(BaseModel):
    id: int
    filename: str
    fake_probability: float
    status: str
    created_at: datetime