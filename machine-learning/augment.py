"""Robustness augmentation for the v2 detector.

Two separate jobs, deliberately kept apart:

``build_train_transform``
    Aggressive *degradation* augmentation. A deployed detector sees faces that
    have been through a video codec, a resize, a webcam ISP and a re-upload,
    while every training corpus is near-pristine PNG. Low-level forgery cues
    (blending seams, GAN checkerboard, upsampling artifacts) are exactly the
    cues destroyed by that pipeline, so a detector trained on clean crops
    latches onto cues that do not survive deployment. Training through the same
    degradations forces it onto cues that do.

``build_eval_transform`` / ``build_corruption``
    Deterministic preprocessing, and a named corruption sweep used by
    ``evaluate_v2.py`` to report AUC as a function of JPEG quality, downscale
    factor and blur radius.

Implemented on ``albumentations`` over uint8 HWC arrays, which is where the
JPEG/downscale operators are cheap and faithful.
"""

from __future__ import annotations

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2

IMAGE_SIZE = 224


def build_train_transform(
    size: int = IMAGE_SIZE,
    mean: tuple[float, ...] = (0.485, 0.456, 0.406),
    std: tuple[float, ...] = (0.229, 0.224, 0.225),
    strength: str = "heavy",
) -> A.Compose:
    """Geometric + photometric + codec augmentation.

    ``strength='none'`` gives resize/flip only (the v1 recipe, for ablation),
    ``'light'`` adds mild photometric jitter, ``'heavy'`` adds the compression
    and resampling chain that deployment actually applies, and ``'heavy_wide'``
    extends that chain down to the quality levels where ``'heavy'`` models were
    measured to break.

    The split between ``heavy`` and ``heavy_wide`` exists because a model
    trained at ``heavy`` scored 0.82 held-out AUC clean and still 0.69 at JPEG
    q10 — but its false-positive rate at q10 jumped from 0.09 to 0.49. Ranking
    survived; calibration did not, because q10 is outside the (30, 95) quality
    range it ever saw. ``heavy_wide`` pushes the floor below the worst quality
    expected at inference so the operating point stays put.
    """
    if strength not in {"none", "light", "heavy", "heavy_wide"}:
        raise ValueError(f"unknown strength '{strength}'")
    wide = strength == "heavy_wide"

    steps: list[A.BasicTransform] = []

    if strength == "none":
        steps += [A.Resize(size, size), A.HorizontalFlip(p=0.5)]
    else:
        # Framing jitter, in both directions. The training crops all come from
        # one alignment pipeline, while the backend crops with MTCNN — measured
        # at roughly 1.4x the detector box, but only to within ±0.1 across
        # frames. Zooming out (scale < 1, reflect-padded) as well as in makes
        # the model tolerate that residual mismatch instead of treating the
        # framing itself as a feature.
        steps += [
            A.Affine(
                scale=(0.8, 1.18),
                translate_percent=(-0.07, 0.07),
                rotate=(-10, 10),
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.8,
            ),
            A.RandomResizedCrop(
                size=(size, size), scale=(0.75, 1.0), ratio=(0.88, 1.14), p=1.0
            ),
            A.HorizontalFlip(p=0.5),
        ]

    if strength in {"light", "heavy", "heavy_wide"}:
        steps += [
            A.RandomBrightnessContrast(
                brightness_limit=0.25, contrast_limit=0.25, p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=8, sat_shift_limit=25, val_shift_limit=15, p=0.3
            ),
            A.ToGray(p=0.05),
        ]

    if strength in {"heavy", "heavy_wide"}:
        steps += [
            # The deployment chain: re-encode, rescale, soften, add sensor noise.
            # One-of blocks keep the expected degradation per sample realistic
            # instead of stacking every operator at once.
            A.ImageCompression(quality_range=(12, 95) if wide else (30, 95), p=0.7),
            A.OneOf(
                [
                    A.Downscale(
                        scale_range=(0.2, 0.8) if wide else (0.35, 0.8), p=1.0
                    ),
                    A.Resize(size // 2, size // 2, p=1.0),
                ],
                p=0.35,
            ),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    A.MotionBlur(blur_limit=(3, 7), p=1.0),
                    A.Sharpen(alpha=(0.1, 0.4), p=1.0),
                ],
                p=0.3,
            ),
            A.GaussNoise(std_range=(0.02, 0.10), p=0.25),
            # A second compression pass mimics re-uploads (the common case for
            # anything that circulated on social media before reaching us).
            A.ImageCompression(quality_range=(20, 95) if wide else (40, 95), p=0.3),
            A.CoarseDropout(
                num_holes_range=(1, 3),
                hole_height_range=(0.05, 0.18),
                hole_width_range=(0.05, 0.18),
                p=0.2,
            ),
        ]

    steps += [A.Resize(size, size), A.Normalize(mean=mean, std=std), ToTensorV2()]
    return A.Compose(steps)


def build_eval_transform(
    size: int = IMAGE_SIZE,
    mean: tuple[float, ...] = (0.485, 0.456, 0.406),
    std: tuple[float, ...] = (0.229, 0.224, 0.225),
) -> A.Compose:
    return A.Compose(
        [A.Resize(size, size), A.Normalize(mean=mean, std=std), ToTensorV2()]
    )


# Corruption sweep for the robustness table. Each entry maps a human-readable
# name to the operator applied to the uint8 crop before the eval transform.
CORRUPTIONS: dict[str, A.BasicTransform | None] = {
    "clean": None,
    "jpeg_q70": A.ImageCompression(quality_range=(70, 70), p=1.0),
    "jpeg_q40": A.ImageCompression(quality_range=(40, 40), p=1.0),
    "jpeg_q20": A.ImageCompression(quality_range=(20, 20), p=1.0),
    "jpeg_q10": A.ImageCompression(quality_range=(10, 10), p=1.0),
    "downscale_0.5": A.Downscale(scale_range=(0.5, 0.5), p=1.0),
    "downscale_0.25": A.Downscale(scale_range=(0.25, 0.25), p=1.0),
    "blur_3": A.GaussianBlur(blur_limit=(3, 3), sigma_limit=(1.0, 1.0), p=1.0),
    "blur_7": A.GaussianBlur(blur_limit=(7, 7), sigma_limit=(2.0, 2.0), p=1.0),
    "noise": A.GaussNoise(std_range=(0.08, 0.08), p=1.0),
    "bright": A.RandomBrightnessContrast(
        brightness_limit=(0.3, 0.3), contrast_limit=(0.0, 0.0), p=1.0
    ),
    "saturate": A.HueSaturationValue(
        hue_shift_limit=0, sat_shift_limit=(40, 40), val_shift_limit=0, p=1.0
    ),
}


def build_corruption(
    name: str,
    size: int = IMAGE_SIZE,
    mean: tuple[float, ...] = (0.485, 0.456, 0.406),
    std: tuple[float, ...] = (0.229, 0.224, 0.225),
) -> A.Compose:
    """Eval transform with one named corruption applied first."""
    if name not in CORRUPTIONS:
        raise ValueError(f"unknown corruption '{name}' — choose from {sorted(CORRUPTIONS)}")
    corruption = CORRUPTIONS[name]
    steps: list[A.BasicTransform] = [] if corruption is None else [corruption]
    steps += [A.Resize(size, size), A.Normalize(mean=mean, std=std), ToTensorV2()]
    return A.Compose(steps)


def apply(transform: A.Compose, image: np.ndarray):
    """``transform(image=...)`` with the albumentations dict unwrapped."""
    return transform(image=image)["image"]
