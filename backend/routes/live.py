import logging

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from .. import database
from ..schemas import LiveFrameOut, LiveSessionOut, LiveStartOut
from ..services import live_session, video_processor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/live", tags=["live"])

MAX_FRAME_BYTES = 4 * 1024 * 1024

INSERT_SESSION = """
INSERT INTO live_sessions (
    started_at, ended_at, duration_seconds, frames_scored,
    mean_probability, peak_probability, status, timeline
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
RETURNING *
"""
SELECT_SESSION = "SELECT * FROM live_sessions WHERE id = %s"


async def read_body(request):
    chunks = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > MAX_FRAME_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"frame exceeds the {MAX_FRAME_BYTES // 1024} KB limit",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def decode_frame(data):
    frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


async def read_rgb(request):
    data = await read_body(request)
    if not data:
        raise HTTPException(status_code=400, detail="request body is empty")
    rgb = decode_frame(data)
    if rgb is None:
        raise HTTPException(status_code=400, detail="request body is not a decodable image")
    return rgb


def load_session(session_id):
    session = live_session.get(session_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail=f"live session {session_id} is not running (it may have timed out)",
        )
    return session


@router.post("/frame", response_model=LiveFrameOut)
async def score_live_frame(request: Request):
    rgb = await read_rgb(request)

    try:
        score = await run_in_threadpool(video_processor.score_frame, rgb)
    except Exception:
        logger.exception("live frame scoring failed")
        raise HTTPException(status_code=500, detail="frame scoring failed")

    if score is None:
        return LiveFrameOut(face_found=False, fake_probability=None, status=None)

    return LiveFrameOut(
        face_found=True,
        fake_probability=score.fake_probability,
        status=video_processor.risk_status(score.fake_probability),
    )


@router.post("/start", response_model=LiveStartOut, status_code=201)
def start_session():
    session = live_session.create()
    logger.info("live session %s started (%d active)", session.id, live_session.active_count())
    return LiveStartOut(session_id=session.id)


@router.post("/{session_id}/frame", response_model=LiveFrameOut)
async def score_session_frame(session_id: str, request: Request):
    session = load_session(session_id)
    rgb = await read_rgb(request)

    if not session.take_slot():
        return LiveFrameOut(**session.verdict(face_found=False, probability=None, dropped=True))

    try:
        verdict = await run_in_threadpool(session.score, rgb)
    except Exception:
        logger.exception("live frame scoring failed for session %s", session_id)
        raise HTTPException(status_code=500, detail="frame scoring failed")
    finally:
        session.release_slot()

    return LiveFrameOut(**verdict)


@router.post("/{session_id}/stop", response_model=LiveSessionOut)
def stop_session(session_id: str):
    session = live_session.finish(session_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail=f"live session {session_id} is not running (it may have timed out)",
        )

    summary = session.summary()
    if not summary["frames_scored"]:
        raise HTTPException(
            status_code=422,
            detail="no face was scored during this session, so there is nothing to record",
        )

    row = database.fetch_one(
        INSERT_SESSION,
        (
            summary["started_at"],
            summary["ended_at"],
            summary["duration_seconds"],
            summary["frames_scored"],
            summary["mean_probability"],
            summary["peak_probability"],
            summary["status"],
            database.to_json(summary["timeline"]),
        ),
    )
    logger.info(
        "live session %s stopped: %d frames, mean %.4f, %s",
        session_id,
        summary["frames_scored"],
        summary["mean_probability"],
        summary["status"],
    )
    return row


@router.get("/sessions/{session_id}", response_model=LiveSessionOut)
def get_session(session_id: int):
    row = database.fetch_one(SELECT_SESSION, (session_id,))
    if row is None:
        raise HTTPException(status_code=404, detail=f"live session {session_id} not found")
    return row
