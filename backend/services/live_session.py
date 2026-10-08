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

    def __init__(self, session_id):
        self.id = session_id
        self.started_at = utcnow()
        self.touched_at = time.monotonic()

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


    def take_slot(self):
        return self._busy.acquire(blocking=False)

    def release_slot(self):
        self._busy.release()


    def face_crop(self, rgb):
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


    def smooth(self, probability):
        alpha = config.LIVE_EMA_ALPHA
        if self.ema is None:
            self.ema = probability
        else:
            self.ema = alpha * probability + (1 - alpha) * self.ema
        return self.ema

    def next_status(self, value):
        gap = config.RISK_HYSTERESIS
        bands = detector.risk_bands()
        high = bands["risk_high"]
        suspicious = bands["risk_suspicious"]
        if self.status == "HIGH_RISK":
            if value >= high - gap:
                return "HIGH_RISK"
        elif value >= high:
            return "HIGH_RISK"

        if self.status in ("SUSPICIOUS", "HIGH_RISK"):
            if value >= suspicious - gap:
                return "SUSPICIOUS"
        elif value >= suspicious:
            return "SUSPICIOUS"

        return "REAL"


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
            self.timeline = self.timeline[::2]

    def score(self, rgb):
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
            "status": video_processor.risk_status(mean),
            "timeline": list(self.timeline),
        }


_sessions = {}
_lock = threading.Lock()


def reap(now=None):
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
    with _lock:
        return _sessions.pop(session_id, None)


def active_count():
    with _lock:
        return len(_sessions)
