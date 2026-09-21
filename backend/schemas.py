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
    user_feedback: str | None
    has_video: bool


class LiveFrameOut(BaseModel):
    face_found: bool
    fake_probability: float | None
    status: str | None
    # Everything below is null on the stateless endpoint, which has no session
    # to smooth against.
    smoothed: float | None = None
    frames_scored: int | None = None
    frames_received: int | None = None
    dropped: bool = False
    recent: list[FrameScore] = []


class LiveStartOut(BaseModel):
    session_id: str


class LiveSessionOut(BaseModel):
    id: int
    started_at: datetime
    ended_at: datetime
    duration_seconds: float
    frames_scored: int
    mean_probability: float
    peak_probability: float
    status: str
    timeline: list[FrameScore]
    user_feedback: str | None


class FeedbackIn(BaseModel):
    label: str | None


class HistoryOut(BaseModel):
    id: int
    # "upload" or "live". Ids are only unique within a kind - they come from
    # two different tables.
    kind: str
    filename: str
    fake_probability: float
    status: str
    created_at: datetime
    has_video: bool
