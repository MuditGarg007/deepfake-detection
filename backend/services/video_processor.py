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
ALIGNED_SCALE = 1.4
ALIGNED_RISE = 0.10
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


def crop_face(frame, box, style=None):
    if style is None:
        style = detector.crop_style()
    h, w = frame.shape[:2]

    if style != "aligned":
        x1, y1, x2, y2 = [int(v) for v in box]
        x1 = max(0, x1 - FACE_MARGIN)
        y1 = max(0, y1 - FACE_MARGIN)
        x2 = min(w, x2 + FACE_MARGIN)
        y2 = min(h, y2 + FACE_MARGIN)
        return frame[y1:y2, x1:x2]

    x1, y1, x2, y2 = [float(v) for v in box]
    side = max(x2 - x1, y2 - y1) * ALIGNED_SCALE
    centre_x = (x1 + x2) / 2
    centre_y = (y1 + y2) / 2 - ALIGNED_RISE * side
    left = int(round(centre_x - side / 2))
    top = int(round(centre_y - side / 2))
    right = int(round(centre_x + side / 2))
    bottom = int(round(centre_y + side / 2))

    region = frame[max(0, top):min(h, bottom), max(0, left):min(w, right)]
    if region.size == 0:
        return None
    pad = (max(0, -top), max(0, bottom - h), max(0, -left), max(0, right - w))
    if any(pad):
        region = cv2.copyMakeBorder(region, *pad, cv2.BORDER_REFLECT_101)
    return region


def detect_box(rgb, face_detector=None):
    face_detector = face_detector or get_mtcnn()
    boxes, probs = face_detector.detect(rgb)
    if boxes is None or len(boxes) == 0:
        return None
    best = largest_face(boxes, probs)
    if best is None:
        return None
    return boxes[best]


def detect_face(rgb, face_detector=None):
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
    bands = detector.risk_bands()
    if prob < bands["risk_suspicious"]:
        return "REAL"
    if prob <= bands["risk_high"]:
        return "SUSPICIOUS"
    return "HIGH_RISK"


def suspicious_region(scores):
    threshold = detector.risk_bands()["frame_threshold"]
    best = []
    current = []
    for score in scores:
        if score.fake_probability >= threshold:
            current.append(score)
            if len(current) > len(best):
                best = current
        else:
            current = []
    if len(best) < MIN_RUN_FRAMES:
        return None, None
    return best[0].timestamp, best[-1].timestamp


def score_frame(rgb, timestamp=0.0, face_detector=None):
    crop = detect_face(rgb, face_detector)
    if crop is None:
        return None
    return FrameScore(timestamp, round(detector.predict(crop), 4))


def aggregate(scores):
    fake_probability = sum(score.fake_probability for score in scores) / len(scores)
    start, end = suspicious_region(scores)
    return round(fake_probability, 4), risk_status(fake_probability), start, end


def process_video(path, filename=None):
    frames = extract_frames(path)
    if not frames:
        raise UnreadableVideoError("video contains no readable frames")

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
