"""Video processing service (T6).

Mirrors Phase 2/3 of docs/plan.md:

1. Extract frames at ~5 fps with cv2, capped at ~100 frames (uniform sample if longer).
2. Detect faces per frame with MTCNN (same approach as Phase 1 preprocessing);
   frames with no face are skipped.
3. Crop the highest-confidence face, resize to 224x224 with a ~20 px margin
   (matching training) — ``detector.predict`` applies the final normalization.
4. Score each crop -> list of ``{timestamp, fake_probability}``.
5. Aggregate: video ``fake_probability`` = mean of frame probs.
6. Risk status: < 0.4 -> REAL, 0.4-0.7 -> SUSPICIOUS, > 0.7 -> HIGH_RISK.
7. Suspicious region: longest contiguous run of frames with
   ``fake_probability >= FRAME_THRESHOLD`` -> (start, end) seconds,
   or None if no run of >= 3 frames.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2

from . import detector
from .config import settings

logger = logging.getLogger(__name__)

SAMPLE_FPS = 5
MAX_FRAMES = 100
CROP_SIZE = 224
FACE_MARGIN = 20
MIN_RUN_FRAMES = 3

# Lazy-loaded MTCNN so the module imports without torch/facenet present.
_mtcnn = None


def _get_mtcnn():
    global _mtcnn
    if _mtcnn is None:
        from facenet_pytorch import MTCNN

        _mtcnn = MTCNN(keep_all=True, min_face_size=40)
    return _mtcnn


def _uniform_sample(total_frames: int, cap: int) -> list[int]:
    """Return up to ``cap`` frame indices spread uniformly across the video."""
    if total_frames <= cap:
        return list(range(total_frames))
    return [round(i * (total_frames - 1) / (cap - 1)) for i in range(cap)]


def _extract_frames(path: Path) -> list[tuple[float, object]]:
    """Return [(timestamp_s, bgr_frame), ...] at ~5 fps, capped at ~100 frames."""
    cap = cv2.VideoCapture(str(path))
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if total <= 0 or fps <= 0:
            raise ValueError("could not read frame count or fps from video")
    finally:
        cap.release()

    sampled = _uniform_sample(total, MAX_FRAMES)
    frames: list[tuple[float, object]] = []
    cap = cv2.VideoCapture(str(path))
    try:
        for idx in sampled:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok:
                frames.append((idx / fps, frame))
    finally:
        cap.release()
    return frames


def _crop_face(frame, box: tuple[float, float, float, float]) -> object:
    """Crop the face box with a ~20 px margin (clamped to the frame)."""
    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = frame.shape[:2]
    x1 = max(0, x1 - FACE_MARGIN)
    y1 = max(0, y1 - FACE_MARGIN)
    x2 = min(w, x2 + FACE_MARGIN)
    y2 = min(h, y2 + FACE_MARGIN)
    return frame[y1:y2, x1:x2]


@dataclass
class FrameScore:
    timestamp: float
    fake_probability: float


@dataclass
class VideoResult:
    filename: str
    fake_probability: float
    status: str
    suspicious_start: float | None
    suspicious_end: float | None
    frame_scores: list[FrameScore]


def _risk_status(prob: float) -> str:
    if prob < settings.RISK_SUSPICIOUS:
        return "REAL"
    if prob <= settings.RISK_HIGH:
        return "SUSPICIOUS"
    return "HIGH_RISK"


def _suspicious_region(
    scores: list[FrameScore],
) -> tuple[float | None, float | None]:
    """Longest contiguous run of frames at/above FRAME_THRESHOLD (>= 3 frames)."""
    best: list[FrameScore] = []
    current: list[FrameScore] = []
    for score in scores:
        if score.fake_probability >= settings.FRAME_THRESHOLD:
            current.append(score)
            if len(current) > len(best):
                best = current
        else:
            current = []
    if len(best) < MIN_RUN_FRAMES:
        return None, None
    return best[0].timestamp, best[-1].timestamp


def process_video(path: Path, filename: str | None = None) -> VideoResult:
    """Run the full pipeline on a video file and return the aggregated result."""
    frames = _extract_frames(path)
    if not frames:
        raise ValueError("video contains no readable frames")

    mtcnn = _get_mtcnn()
    scores: list[FrameScore] = []
    for timestamp, frame in frames:
        bgr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, probs = mtcnn.detect(bgr)
        if boxes is None or len(boxes) == 0:
            continue
        # Highest-confidence face, first if ties.
        best = max(range(len(boxes)), key=lambda i: probs[i])
        crop = _crop_face(bgr, boxes[best])
        prob = detector.predict(crop)
        scores.append(FrameScore(timestamp=timestamp, fake_probability=prob))

    if not scores:
        raise ValueError("no faces detected in any frame")

    fake_probability = sum(s.fake_probability for s in scores) / len(scores)
    start, end = _suspicious_region(scores)
    return VideoResult(
        filename=filename or Path(path).name,
        fake_probability=round(fake_probability, 4),
        status=_risk_status(fake_probability),
        suspicious_start=start,
        suspicious_end=end,
        frame_scores=scores,
    )
