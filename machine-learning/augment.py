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
    if strength not in {"none", "light", "heavy", "heavy_wide"}:
        raise ValueError(f"unknown strength '{strength}'")
    wide = strength == "heavy_wide"

    steps: list[A.BasicTransform] = []

    if strength == "none":
        steps += [A.Resize(size, size), A.HorizontalFlip(p=0.5)]
    else:
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
    if name not in CORRUPTIONS:
        raise ValueError(f"unknown corruption '{name}' — choose from {sorted(CORRUPTIONS)}")
    corruption = CORRUPTIONS[name]
    steps: list[A.BasicTransform] = [] if corruption is None else [corruption]
    steps += [A.Resize(size, size), A.Normalize(mean=mean, std=std), ToTensorV2()]
    return A.Compose(steps)


def apply(transform: A.Compose, image: np.ndarray):
    return transform(image=image)["image"]
