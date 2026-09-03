"""T4 — dataset + dataloaders over the preprocessed face crops."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMAGE_SIZE = 224
# ImageNet statistics — required because the backbones are ImageNet-pretrained.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
LABEL_TO_TARGET = {"real": 0.0, "fake": 1.0}


def build_transform(train: bool) -> transforms.Compose:
    """Train-time augmentation (roadmap T4) or plain eval preprocessing."""
    steps: list = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
    if train:
        steps += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    steps += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(steps)


class FaceDataset(Dataset):
    """Face crops for one split, read from ``manifest.csv``.

    Yields ``(image_tensor, target)`` where target is 1.0 for fake, 0.0 for real.
    """

    def __init__(self, data_dir: str | Path, split: str, train: bool | None = None):
        self.data_dir = Path(data_dir)
        self.split = split
        manifest = self.data_dir / "manifest.csv"
        if not manifest.is_file():
            raise FileNotFoundError(
                f"{manifest} not found — run preprocessing.py first"
            )
        frame = pd.read_csv(manifest)
        frame = frame[frame["split"] == split].reset_index(drop=True)
        if frame.empty:
            raise ValueError(f"manifest has no rows for split '{split}'")
        self.frame = frame
        self.transform = build_transform(train if train is not None else split == "train")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.frame.iloc[index]
        image = Image.open(self.data_dir / row["path"]).convert("RGB")
        target = torch.tensor(LABEL_TO_TARGET[row["label"]], dtype=torch.float32)
        return self.transform(image), target

    def class_counts(self) -> dict[str, int]:
        return self.frame["label"].value_counts().to_dict()


def build_dataloaders(
    data_dir: str | Path,
    batch_size: int = 64,
    num_workers: int = 4,
    splits: tuple[str, ...] = ("train", "val", "test"),
) -> dict[str, DataLoader]:
    """DataLoader per split; only the train loader shuffles and augments."""
    loaders: dict[str, DataLoader] = {}
    for split in splits:
        dataset = FaceDataset(data_dir, split)
        is_train = split == "train"
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=is_train,
            persistent_workers=num_workers > 0,
        )
    return loaders
