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

# A 640px-wide JPEG at quality 0.95 is well under 100 KB; the cap is only here
# so a misdirected request cannot buffer something large.
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
    """Read the request body, refusing anything over MAX_FRAME_BYTES."""
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
    """Decode image bytes into an RGB array, or None if they are not an image."""
    frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


async def read_rgb(request):
    """Read one request body and decode it, or raise the right 4xx."""
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
    """Score a single JPEG frame. Stateless: nothing is stored, nothing is smoothed.

    The session endpoints below are what the UI uses; this one stays for
    one-shot checks against a still.

    The frame is held in memory for the length of the request and then dropped -
    live frames are never written to UPLOAD_DIR, and the bytes never reach the log.
    """
    rgb = await read_rgb(request)

    try:
        # Detection plus inference is 150-400 ms of CPU, so keep it off the loop.
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
    """Mint a session id. State lives in this process until /stop or a timeout."""
    session = live_session.create()
    logger.info("live session %s started (%d active)", session.id, live_session.active_count())
    return LiveStartOut(session_id=session.id)


@router.post("/{session_id}/frame", response_model=LiveFrameOut)
async def score_session_frame(session_id: str, request: Request):
    """Score one frame into a running session and return the smoothed verdict."""
    session = load_session(session_id)
    rgb = await read_rgb(request)

    # Single slot: a frame that arrives while the last one is still scoring is
    # dropped, and the caller gets the verdict that currently stands.
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
    """Finalise a session and write its one summary row.

    Written once, on stop: per-frame inserts at 2 fps would put thousands of
    round trips in the hot path for nothing. The row holds scores, never
    images - a live session has no stored frame to point at.
    """
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
    """Read back a finished session by its row id."""
    row = database.fetch_one(SELECT_SESSION, (session_id,))
    if row is None:
        raise HTTPException(status_code=404, detail=f"live session {session_id} not found")
    return row
