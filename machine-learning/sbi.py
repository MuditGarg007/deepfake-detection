from __future__ import annotations

import cv2
import numpy as np

HULL_POINTS = list(range(0, 17)) + list(range(68, 81))


def _rand(low: float, high: float, rng: np.random.Generator) -> float:
    return float(rng.uniform(low, high))


def convex_hull_mask(
    landmarks: np.ndarray, shape: tuple[int, int], rng: np.random.Generator
) -> np.ndarray:
    height, width = shape
    points = landmarks[HULL_POINTS].astype(np.float32)

    points = points + rng.normal(0, _rand(1.0, 5.0, rng), points.shape)
    centre = points.mean(axis=0, keepdims=True)
    points = centre + (points - centre) * _rand(0.88, 1.12, rng)

    mask = np.zeros((height, width), dtype=np.uint8)
    hull = cv2.convexHull(points.astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 255)

    size = int(_rand(3, 17, rng)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    mask = cv2.dilate(mask, kernel) if rng.random() < 0.5 else cv2.erode(mask, kernel)
    blur = int(_rand(5, 33, rng)) | 1
    mask = cv2.GaussianBlur(mask, (blur, blur), 0)

    alpha = mask.astype(np.float32) / 255.0
    return (alpha * _rand(0.6, 1.0, rng))[..., None]


def source_transform(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = image.astype(np.float32)

    if rng.random() < 0.7:
        out = out * rng.uniform(0.92, 1.08, size=(1, 1, 3))
        out = out + rng.uniform(-11, 11, size=(1, 1, 3))

    if rng.random() < 0.5:
        out = (out - 128.0) * _rand(0.9, 1.12, rng) + 128.0 + _rand(-9, 9, rng)

    out = np.clip(out, 0, 255).astype(np.uint8)

    if rng.random() < 0.5:
        scale = _rand(0.5, 0.95, rng)
        height, width = out.shape[:2]
        small = cv2.resize(out, (max(int(width * scale), 8), max(int(height * scale), 8)),
                           interpolation=cv2.INTER_AREA)
        out = cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)

    if rng.random() < 0.3:
        size = int(_rand(3, 7, rng)) | 1
        out = cv2.GaussianBlur(out, (size, size), 0)

    if rng.random() < 0.4:
        quality = int(_rand(40, 95, rng))
        ok, buffer = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if ok:
            out = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

    return out


def geometric_jitter(
    image: np.ndarray, landmarks: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    height, width = image.shape[:2]
    angle = _rand(-4, 4, rng)
    scale = _rand(0.96, 1.04, rng)
    centre = tuple(landmarks.mean(axis=0).astype(np.float32))
    matrix = cv2.getRotationMatrix2D(centre, angle, scale)
    matrix[0, 2] += _rand(-6, 6, rng)
    matrix[1, 2] += _rand(-6, 6, rng)
    return cv2.warpAffine(
        image, matrix, (width, height), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def self_blend(
    image: np.ndarray, landmarks: np.ndarray, rng: np.random.Generator | None = None
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    if landmarks is None or len(landmarks) < 81:
        return image

    height, width = image.shape[:2]
    landmarks = np.clip(
        landmarks.astype(np.float32), [0, 0], [width - 1, height - 1]
    )

    target = image
    source = source_transform(image, rng)
    if rng.random() < 0.8:
        source = geometric_jitter(source, landmarks, rng)

    if rng.random() < 0.3:
        source, target = target, source

    alpha = convex_hull_mask(landmarks, (height, width), rng)
    blended = source.astype(np.float32) * alpha + target.astype(np.float32) * (1 - alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)
