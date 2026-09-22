"""Self-Blended Images — pseudo-fakes manufactured from genuine faces.

After Shiohara & Yamasaki, "Detecting Deepfakes with Self-Blended Images"
(CVPR 2022). The idea is worth the implementation cost here because it attacks
the exact failure this project has:

Almost every face swap — DeepFaceLab, SimSwap, InSwapper, FaceDancer, whatever
ships next year — ends by compositing a generated face region onto an original
frame. The generator differs wildly between methods; the *composite* does not.
It always leaves a blending boundary and a statistical discontinuity between
the pasted region and its surroundings. A detector trained on one generator
learns that generator; a detector trained on blending boundaries learns the
step every generator shares.

SBI synthesises those boundaries from a single real image: take two
differently-degraded copies of one face, mildly warp one of them, and blend
along a deformed convex hull of the facial landmarks. No second identity and no
generator is involved, so the pseudo-fake cannot leak any method-specific
artifact — which is precisely why it transfers.

Used here *alongside* real manipulated data rather than instead of it (the
paper trains on SBI alone). Entire-face-synthesis forgeries — StyleGAN,
diffusion, MidJourney — have no blending boundary at all, so SBI cannot cover
them; the DF40 synthesis methods do.
"""

from __future__ import annotations

import cv2
import numpy as np

# dlib-81 landmark contour used as the blend region. Points 0-16 trace the jaw;
# 68-80 are the forehead points DeepfakeBench's 81-point model adds, which is
# what lets the hull cover the whole face rather than stopping at the brows.
HULL_POINTS = list(range(0, 17)) + list(range(68, 81))


def _rand(low: float, high: float, rng: np.random.Generator) -> float:
    return float(rng.uniform(low, high))


def convex_hull_mask(
    landmarks: np.ndarray, shape: tuple[int, int], rng: np.random.Generator
) -> np.ndarray:
    """Soft face mask from landmarks, randomly deformed.

    The deformation matters: a clean hull would teach the model one fixed mask
    shape, and real swaps all use different (and usually worse) masks.
    """
    height, width = shape
    points = landmarks[HULL_POINTS].astype(np.float32)

    # Random per-point jitter, then shrink/grow the hull about its centroid, so
    # the boundary lands somewhere plausible rather than exactly on the jawline.
    points = points + rng.normal(0, _rand(1.0, 5.0, rng), points.shape)
    centre = points.mean(axis=0, keepdims=True)
    points = centre + (points - centre) * _rand(0.88, 1.12, rng)

    mask = np.zeros((height, width), dtype=np.uint8)
    hull = cv2.convexHull(points.astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 255)

    # Erode or dilate, then feather. The feather width controls how visible the
    # seam is; sampling it wide covers everything from a crude paste to a
    # carefully poisson-blended swap.
    size = int(_rand(3, 17, rng)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    mask = cv2.dilate(mask, kernel) if rng.random() < 0.5 else cv2.erode(mask, kernel)
    blur = int(_rand(5, 33, rng)) | 1
    mask = cv2.GaussianBlur(mask, (blur, blur), 0)

    alpha = mask.astype(np.float32) / 255.0
    return (alpha * _rand(0.6, 1.0, rng))[..., None]


def source_transform(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Photometric/frequency mismatch between the pasted region and the frame.

    A swap's generated region almost never matches the host frame's colour
    balance, sharpness or compression history. These are the differences the
    detector should key on, so they are introduced deliberately.
    """
    out = image.astype(np.float32)

    # Magnitudes are kept modest on purpose. A violent colour shift produces a
    # pseudo-fake no real swap resembles, and the model then learns "weird
    # colour = fake" instead of "the face and the frame disagree".
    if rng.random() < 0.7:  # colour / contrast shift
        out = out * rng.uniform(0.92, 1.08, size=(1, 1, 3))
        out = out + rng.uniform(-11, 11, size=(1, 1, 3))

    if rng.random() < 0.5:  # brightness+contrast around the midpoint
        out = (out - 128.0) * _rand(0.9, 1.12, rng) + 128.0 + _rand(-9, 9, rng)

    out = np.clip(out, 0, 255).astype(np.uint8)

    if rng.random() < 0.5:  # resolution mismatch
        scale = _rand(0.5, 0.95, rng)
        height, width = out.shape[:2]
        small = cv2.resize(out, (max(int(width * scale), 8), max(int(height * scale), 8)),
                           interpolation=cv2.INTER_AREA)
        out = cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)

    if rng.random() < 0.3:  # sharpness mismatch
        size = int(_rand(3, 7, rng)) | 1
        out = cv2.GaussianBlur(out, (size, size), 0)

    if rng.random() < 0.4:  # compression-history mismatch
        quality = int(_rand(40, 95, rng))
        ok, buffer = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if ok:
            out = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

    return out


def geometric_jitter(
    image: np.ndarray, landmarks: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Sub-pixel-to-few-pixel misalignment of the source face.

    Swaps land the generated face slightly off the original's geometry. Without
    this the blend is pixel-perfect inside the mask and the only cue left is the
    seam itself.
    """
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
    """Blend a degraded, slightly-warped copy of ``image`` back into itself.

    ``image`` is RGB uint8 HWC, ``landmarks`` is the (81, 2) array saved
    alongside it. Returns an RGB uint8 pseudo-fake of the same size.
    """
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

    # Occasionally degrade the frame instead of the face, so the cue the model
    # learns is "these two regions disagree" rather than the fixed polarity
    # "the masked region is the blurrier one". Kept well below half the time:
    # a real swap always alters the face, so that is the dominant case.
    if rng.random() < 0.3:
        source, target = target, source

    alpha = convex_hull_mask(landmarks, (height, width), rng)
    blended = source.astype(np.float32) * alpha + target.astype(np.float32) * (1 - alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)
