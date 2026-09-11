"""Video processing service (T6).

Mirrors Phase 2/3 of docs/plan.md:

1. Extract frames at ~5 fps with cv2, capped at ~100 frames (uniform sample if longer).
2. Detect faces per frame with MTCNN (same approach as Phase 1 preprocessing);
   frames with no face are skipped.
3. Crop the largest face detected above ``FACE_CONF`` with a ~20 px margin, the
   same selection rule Phase 1 preprocessing used to build the training crops.
   ``detector.predict`` applies the resize to 224x224 and the normalization.
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

from ..config import settings
from . import detector

logger = logging.getLogger(__name__)

class VideoProcessingError(Exception):
    """Base class for videos the pipeline cannot produce a result for."""


class UnreadableVideoError(VideoProcessingError):
    """The file is not a readable video (no frames, no fps)."""


class NoFacesError(VideoProcessingError):
    """The video is readable but contains no detectable face."""


SAMPLE_FPS = 5
MAX_FRAMES = 100
FACE_MARGIN = 20
# Minimum MTCNN confidence, matching preprocessing.py's --conf default.
FACE_CONF = 0.95
MIN_RUN_FRAMES = 3

# Lazy-loaded MTCNN so the module imports without torch/facenet present.
_mtcnn = None


def _get_mtcnn():
    global _mtcnn
    if _mtcnn is None:
        from facenet_pytorch import MTCNN

        _mtcnn = MTCNN(keep_all=True, min_face_size=40)
    return _mtcnn


def _sample_indices(total_frames: int, src_fps: float, cap: int) -> list[int]:
    """Frame indices at ``SAMPLE_FPS``, uniformly thinned to at most ``cap``.

    Same two-stage sampling as Phase 1 preprocessing, so the frames scored here
    are drawn the same way as the frames the model was trained on.
    """
    step = max(1, round(src_fps / SAMPLE_FPS)) if src_fps > 0 else 1
    indices = list(range(0, total_frames, step))
    if len(indices) > cap:
        picks = [round(i * (len(indices) - 1) / (cap - 1)) for i in range(cap)]
        indices = [indices[p] for p in dict.fromkeys(picks)]
    return indices


def _extract_frames(path: Path) -> list[tuple[float, object]]:
    """Return [(timestamp_s, bgr_frame), ...] at ~5 fps, capped at ~100 frames."""
    cap = cv2.VideoCapture(str(path))
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if total <= 0 or fps <= 0:
            raise UnreadableVideoError("could not read frame count or fps from video")
    finally:
        cap.release()

    sampled = _sample_indices(total, fps, MAX_FRAMES)
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


def _largest_face(boxes, probs) -> int | None:
    """Index of the largest box among detections above ``FACE_CONF``, or None.

    Phase 1 built the training crops this way; picking by confidence instead
    would feed the model differently sized faces than it was trained on.
    """
    keep = [i for i, p in enumerate(probs) if p is not None and p >= FACE_CONF]
    if not keep:
        return None
    return max(keep, key=lambda i: (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]))


def _crop_face(frame, box: tuple[float, float, float, float]) -> object:
    """Crop the face box with a ~20 px margin (clamped to the frame)."""
    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = frame.shape[:2]
    x1 = max(0, x1 - FACE_MARGIN)
    y1 = max(0, y1 - FACE_MARGIN)
    x2 = min(w, x2 + FACE_MARGIN)
    y2 = min(h, y2 + FACE_MARGIN)
    return frame[y1:y2, x1:x2]


def grab_frame_jpeg(path: Path, timestamp: float, quality: int = 85) -> bytes | None:
    """Return the frame nearest ``timestamp`` as JPEG bytes, or None if unreadable.

    Scored frames are held in memory during ``process_video`` and never written
    to disk, so showing one afterwards means seeking the source video again.
    """
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, round(timestamp * fps))
        ok, frame = cap.read()
        if not ok:
            return None
        encoded, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buffer.tobytes() if encoded else None
    finally:
        cap.release()


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
        raise UnreadableVideoError("video contains no readable frames")

    mtcnn = _get_mtcnn()
    timestamps: list[float] = []
    crops: list[object] = []
    for timestamp, frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, probs = mtcnn.detect(rgb)
        if boxes is None or len(boxes) == 0:
            continue
        best = _largest_face(boxes, probs)
        if best is None:
            continue
        timestamps.append(timestamp)
        crops.append(_crop_face(rgb, boxes[best]))

    if not crops:
        raise NoFacesError("no faces detected in any frame")

    # One batched call instead of one per frame - inference.predict accepts a
    # list and scores it in a single forward pass.
    probabilities = detector.predict_batch(crops)
    scores = [
        FrameScore(timestamp=t, fake_probability=round(p, 4))
        for t, p in zip(timestamps, probabilities)
    ]

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
