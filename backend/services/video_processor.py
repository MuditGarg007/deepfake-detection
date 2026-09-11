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


def process_video(path, filename=None):
    frames = extract_frames(path)
    if not frames:
        raise UnreadableVideoError("video contains no readable frames")

    face_detector = get_mtcnn()
    timestamps = []
    crops = []
    for timestamp, frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, probs = face_detector.detect(rgb)
        if boxes is None or len(boxes) == 0:
            continue
        best = largest_face(boxes, probs)
        if best is None:
            continue
        timestamps.append(timestamp)
        crops.append(crop_face(rgb, boxes[best]))

    if not crops:
        raise NoFacesError("no faces detected in any frame")

    probabilities = detector.predict_batch(crops)
    scores = [
        FrameScore(t, round(p, 4)) for t, p in zip(timestamps, probabilities)
    ]

    fake_probability = sum(s.fake_probability for s in scores) / len(scores)
    start, end = suspicious_region(scores)
    return VideoResult(
        filename=filename or Path(path).name,
        fake_probability=round(fake_probability, 4),
        status=risk_status(fake_probability),
        suspicious_start=start,
        suspicious_end=end,
        frame_scores=scores,
    )
