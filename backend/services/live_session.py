"""In-memory state for one live screen-share.

A live session is a short-lived thing that lives entirely in this process: a
ring buffer of recent scores, a smoothed value the badge reads, and enough
counters to write one summary row when the share stops. Nothing here touches
the database, and no frame bytes are kept - only the numbers they produced.

Because the state is process-local, a multi-instance deployment needs session
affinity; see docs/live-screenshare-refactor.md section 7.
"""

import logging
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone

from .. import config
from . import detector, video_processor

logger = logging.getLogger(__name__)


def utcnow():
    return datetime.now(timezone.utc)


class LiveSession:
    """Scores frames for one share and smooths the result.

    Not safe to score from two threads at once - callers take `slot()` first,
    which is also what enforces the single-slot backpressure policy.
    """

    def __init__(self, session_id):
        self.id = session_id
        self.started_at = utcnow()
        self.touched_at = time.monotonic()

        # Recent smoothed values for the sparkline, and the full run for the
        # stored timeline.
        self.recent = deque(maxlen=config.LIVE_HISTORY_FRAMES)
        self.timeline = []

        self.ema = None
        self.status = "WAITING"
        self.last_box = None
        self.frames_since_detect = 0

        self.frames_received = 0
        self.frames_scored = 0
        self.total = 0.0
        self.peak = 0.0

        self._busy = threading.Lock()

    # --- backpressure --------------------------------------------------------

    def take_slot(self):
        """Claim the single scoring slot, or return False if one is in use.

        A frame that arrives while the previous one is still being scored is
        dropped rather than queued: a queue that grows makes the badge describe
        the past, which is worse than a badge that updates less often.
        """
        return self._busy.acquire(blocking=False)

    def release_slot(self):
        self._busy.release()

    # --- face tracking -------------------------------------------------------

    def face_crop(self, rgb):
        """Crop the face out of one frame, running MTCNN only every Nth call.

        Detection dominates the per-frame cost and a face does not teleport in
        500 ms, so in between runs the previous box is reused.
        """
        if self.last_box is not None and self.frames_since_detect < config.LIVE_DETECT_EVERY:
            self.frames_since_detect += 1
            crop = video_processor.crop_face(rgb, self.last_box)
            if crop.size:
                return crop

        self.frames_since_detect = 0
        self.last_box = video_processor.detect_box(rgb)
        if self.last_box is None:
            return None
        crop = video_processor.crop_face(rgb, self.last_box)
        return crop if crop.size else None

    # --- smoothing -----------------------------------------------------------

    def smooth(self, probability):
        alpha = config.LIVE_EMA_ALPHA
        if self.ema is None:
            self.ema = probability
        else:
            self.ema = alpha * probability + (1 - alpha) * self.ema
        return self.ema

    def next_status(self, value):
        """Risk status with a sticky exit.

        The entry thresholds are the configured ones, so live and upload agree
        on what the numbers mean. Only leaving a state is sticky, by
        RISK_HYSTERESIS - without the gap a score sitting on a threshold makes
        the badge strobe between two states several times a second.
        """
        gap = config.RISK_HYSTERESIS
        if self.status == "HIGH_RISK":
            if value >= config.RISK_HIGH - gap:
                return "HIGH_RISK"
        elif value >= config.RISK_HIGH:
            return "HIGH_RISK"

        if self.status in ("SUSPICIOUS", "HIGH_RISK"):
            if value >= config.RISK_SUSPICIOUS - gap:
                return "SUSPICIOUS"
        elif value >= config.RISK_SUSPICIOUS:
            return "SUSPICIOUS"

        return "REAL"

    # --- scoring -------------------------------------------------------------

    def elapsed(self):
        return (utcnow() - self.started_at).total_seconds()

    def record(self, probability):
        timestamp = round(self.elapsed(), 2)
        smoothed = round(self.smooth(probability), 4)
        self.status = self.next_status(smoothed)

        self.frames_scored += 1
        self.total += probability
        self.peak = max(self.peak, probability)
        self.recent.append({"timestamp": timestamp, "fake_probability": smoothed})

        self.timeline.append({"timestamp": timestamp, "fake_probability": smoothed})
        if len(self.timeline) > config.LIVE_TIMELINE_MAX:
            # Halve the resolution rather than dropping the tail, so a long
            # session keeps its whole shape.
            self.timeline = self.timeline[::2]

    def score(self, rgb):
        """Score one frame and fold it into the session. Returns a verdict dict.

        When no face is found the last verdict is held rather than reset: a
        face that leaves the frame for a moment should not read as REAL.
        """
        self.touched_at = time.monotonic()
        self.frames_received += 1

        crop = self.face_crop(rgb)
        if crop is None:
            return self.verdict(face_found=False, probability=None)

        probability = round(detector.predict(crop), 4)
        self.record(probability)
        return self.verdict(face_found=True, probability=probability)

    def verdict(self, face_found, probability, dropped=False):
        return {
            "face_found": face_found,
            "fake_probability": probability,
            "smoothed": None if self.ema is None else round(self.ema, 4),
            "status": self.status,
            "frames_scored": self.frames_scored,
            "frames_received": self.frames_received,
            "dropped": dropped,
            "recent": list(self.recent),
        }

    # --- summary -------------------------------------------------------------

    def summary(self):
        ended_at = utcnow()
        mean = self.total / self.frames_scored if self.frames_scored else 0.0
        return {
            "started_at": self.started_at,
            "ended_at": ended_at,
            "duration_seconds": round((ended_at - self.started_at).total_seconds(), 2),
            "frames_received": self.frames_received,
            "frames_scored": self.frames_scored,
            "mean_probability": round(mean, 4),
            "peak_probability": round(self.peak, 4),
            # The badge state is smoothed and sticky; the stored verdict is the
            # plain mean read through the shared thresholds, so it means the
            # same thing as an uploaded video's.
            "status": video_processor.risk_status(mean),
            "timeline": list(self.timeline),
        }


# --- registry ---------------------------------------------------------------

_sessions = {}
_lock = threading.Lock()


def reap(now=None):
    """Drop sessions that have gone quiet, so an abandoned tab cannot leak."""
    now = now if now is not None else time.monotonic()
    with _lock:
        stale = [
            key for key, session in _sessions.items()
            if now - session.touched_at > config.LIVE_SESSION_TTL
        ]
        for key in stale:
            del _sessions[key]
    if stale:
        logger.info("reaped %d idle live session(s)", len(stale))
    return stale


def create():
    reap()
    session = LiveSession(uuid.uuid4().hex)
    with _lock:
        _sessions[session.id] = session
    return session


def get(session_id):
    with _lock:
        return _sessions.get(session_id)


def finish(session_id):
    """Remove a session and return it, or None if it was never there."""
    with _lock:
        return _sessions.pop(session_id, None)


def active_count():
    with _lock:
        return len(_sessions)
