import logging
from dataclasses import dataclass
from pathlib import Path

import cv2

from .. import config
from . import detector

logger = logging.getLogger(__name__)


class VideoProcessingError(Exception):
    pass


class UnreadableVideoError(VideoProcessingError):
    pass


class NoFacesError(VideoProcessingError):
    pass


SAMPLE_FPS = 5
MAX_FRAMES = 100
FACE_MARGIN = 20
FACE_CONF = 0.95
MIN_RUN_FRAMES = 3

mtcnn = None


def get_mtcnn():
    global mtcnn
    if mtcnn is None:
        from facenet_pytorch import MTCNN

        mtcnn = MTCNN(keep_all=True, min_face_size=40)
    return mtcnn


def sample_indices(total_frames, src_fps, cap):
    step = max(1, round(src_fps / SAMPLE_FPS)) if src_fps > 0 else 1
    indices = list(range(0, total_frames, step))
    if len(indices) > cap:
        picks = [round(i * (len(indices) - 1) / (cap - 1)) for i in range(cap)]
        indices = [indices[p] for p in dict.fromkeys(picks)]
    return indices


def extract_frames(path):
    cap = cv2.VideoCapture(str(path))
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if total <= 0 or fps <= 0:
            raise UnreadableVideoError("could not read frame count or fps from video")
    finally:
        cap.release()

    sampled = sample_indices(total, fps, MAX_FRAMES)
    frames = []
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


def largest_face(boxes, probs):
    keep = [i for i, p in enumerate(probs) if p is not None and p >= FACE_CONF]
    if not keep:
        return None
    return max(keep, key=lambda i: (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]))


def crop_face(frame, box):
    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = frame.shape[:2]
    x1 = max(0, x1 - FACE_MARGIN)
    y1 = max(0, y1 - FACE_MARGIN)
    x2 = min(w, x2 + FACE_MARGIN)
    y2 = min(h, y2 + FACE_MARGIN)
    return frame[y1:y2, x1:x2]


def detect_box(rgb, face_detector=None):
    """Return the box of the largest confident face in one RGB frame, or None.

    Split out from detect_face so the live path can hold on to a box and reuse
    it for a few frames instead of running MTCNN on every one.
    """
    face_detector = face_detector or get_mtcnn()
    boxes, probs = face_detector.detect(rgb)
    if boxes is None or len(boxes) == 0:
        return None
    best = largest_face(boxes, probs)
    if best is None:
        return None
    return boxes[best]


def detect_face(rgb, face_detector=None):
    """Return the largest confident face crop in one RGB frame, or None.

    Shared by the whole-video path, which batches the crops it collects, and
    by score_frame, which has exactly one frame to work with.
    """
    box = detect_box(rgb, face_detector)
    if box is None:
        return None
    return crop_face(rgb, box)


def grab_frame_jpeg(path, timestamp, quality=85):
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, round(timestamp * fps))
        ok, frame = cap.read()
        if not ok:
            return None
        encoded, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not encoded:
            return None
        return buffer.tobytes()
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
    frame_scores: list


def risk_status(prob):
    if prob < config.RISK_SUSPICIOUS:
        return "REAL"
    if prob <= config.RISK_HIGH:
        return "SUSPICIOUS"
    return "HIGH_RISK"


def suspicious_region(scores):
    best = []
    current = []
    for score in scores:
        if score.fake_probability >= config.FRAME_THRESHOLD:
            current.append(score)
            if len(current) > len(best):
                best = current
        else:
            current = []
    if len(best) < MIN_RUN_FRAMES:
        return None, None
    return best[0].timestamp, best[-1].timestamp


def score_frame(rgb, timestamp=0.0, face_detector=None):
    """Detect the largest face in one RGB frame and score it.

    Returns None when no face clears FACE_CONF.
    """
    crop = detect_face(rgb, face_detector)
    if crop is None:
        return None
    return FrameScore(timestamp, round(detector.predict(crop), 4))


def aggregate(scores):
    """Mean probability, risk status and suspicious region for a run of frames."""
    fake_probability = sum(score.fake_probability for score in scores) / len(scores)
    start, end = suspicious_region(scores)
    return round(fake_probability, 4), risk_status(fake_probability), start, end


def process_video(path, filename=None):
    frames = extract_frames(path)
    if not frames:
        raise UnreadableVideoError("video contains no readable frames")

    # Detection is per frame, but scoring stays batched - predict_batch is
    # meaningfully faster than one call per crop.
    face_detector = get_mtcnn()
    timestamps = []
    crops = []
    for timestamp, frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        crop = detect_face(rgb, face_detector)
        if crop is None:
            continue
        timestamps.append(timestamp)
        crops.append(crop)

    if not crops:
        raise NoFacesError("no faces detected in any frame")

    probabilities = detector.predict_batch(crops)
    scores = [
        FrameScore(t, round(p, 4)) for t, p in zip(timestamps, probabilities)
    ]

    fake_probability, status, start, end = aggregate(scores)
    return VideoResult(
        filename=filename or Path(path).name,
        fake_probability=fake_probability,
        status=status,
        suspicious_start=start,
        suspicious_end=end,
        frame_scores=scores,
    )
