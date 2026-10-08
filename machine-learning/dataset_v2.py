from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from augment import build_eval_transform, build_train_transform
from sbi import self_blend

LABEL_TO_TARGET = {"real": 0.0, "fake": 1.0}
SBI_METHOD = "sbi"

cv2.setNumThreads(0)


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read image {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class ForgeryDataset(Dataset):

    def __init__(
        self,
        data_dir: str | Path,
        manifest: str | Path | pd.DataFrame,
        split: str | None = None,
        transform=None,
        datasets: tuple[str, ...] | None = None,
        methods: tuple[str, ...] | None = None,
        exclude_methods: tuple[str, ...] = (),
        sbi_prob: float = 0.0,
    ):
        self.data_dir = Path(data_dir)
        frame = (
            manifest.copy()
            if isinstance(manifest, pd.DataFrame)
            else pd.read_csv(manifest)
        )
        if split is not None:
            frame = frame[frame["split"] == split]
        if datasets is not None:
            frame = frame[frame["dataset"].isin(datasets)]
        if methods is not None:
            frame = frame[frame["method"].isin(methods)]
        if exclude_methods:
            frame = frame[~frame["method"].isin(exclude_methods)]
        frame = frame.reset_index(drop=True)
        if frame.empty:
            raise ValueError(
                f"manifest selection is empty (split={split}, datasets={datasets}, "
                f"methods={methods}, exclude={exclude_methods})"
            )

        self.frame = frame
        self.transform = transform if transform is not None else build_eval_transform()
        self.sbi_prob = sbi_prob
        self.method_names = sorted(frame["method"].unique())
        if sbi_prob > 0 and SBI_METHOD not in self.method_names:
            self.method_names.append(SBI_METHOD)
        self.method_to_index = {name: i for i, name in enumerate(self.method_names)}
        self._paths = frame["path"].to_numpy()
        self._targets = frame["label"].map(LABEL_TO_TARGET).to_numpy(dtype=np.float32)
        self._methods = frame["method"].map(self.method_to_index).to_numpy(dtype=np.int64)
        landmarks = frame["landmark"] if "landmark" in frame.columns else ""
        self._landmarks = pd.Series(landmarks, index=frame.index).fillna("").to_numpy()

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        image = read_rgb(self.data_dir / self._paths[index])
        target = self._targets[index]
        method = int(self._methods[index])

        landmark = self._landmarks[index]
        if self.sbi_prob > 0 and target == 0.0 and landmark:
            if float(torch.rand(())) < self.sbi_prob:
                path = self.data_dir / landmark
                if path.is_file():
                    seed = int(torch.randint(0, 2**31 - 1, ()))
                    image = self_blend(
                        image, np.load(path), np.random.default_rng(seed)
                    )
                    target = 1.0
                    method = self.method_to_index[SBI_METHOD]

        tensor = self.transform(image=image)["image"]
        return tensor, torch.tensor(target, dtype=torch.float32), method

    def summary(self) -> pd.DataFrame:
        return (
            self.frame.groupby(["dataset", "method", "label"])
            .size()
            .rename("frames")
            .reset_index()
        )


MAX_BALANCEABLE_SBI_PROB = 0.45


def real_mass_for(sbi_prob: float) -> float:
    if sbi_prob <= 0:
        return 0.5
    if sbi_prob >= MAX_BALANCEABLE_SBI_PROB:
        raise ValueError(
            f"sbi_prob={sbi_prob} cannot be class-balanced; keep it below "
            f"{MAX_BALANCEABLE_SBI_PROB}"
        )
    return 1.0 / (2.0 * (1.0 - sbi_prob))


def balanced_weights(frame: pd.DataFrame, sbi_prob: float = 0.0) -> np.ndarray:
    weights = np.zeros(len(frame), dtype=np.float64)

    is_real = (frame["label"] == "real").to_numpy()
    n_real = int(is_real.sum())

    effective_prob = 0.0
    if sbi_prob > 0 and n_real:
        if "landmark" in frame.columns:
            has_landmark = frame["landmark"].fillna("").astype(bool).to_numpy()
            effective_prob = sbi_prob * float((has_landmark & is_real).sum()) / n_real
        else:
            effective_prob = 0.0

    real_mass = real_mass_for(effective_prob)
    if n_real:
        weights[is_real] = real_mass / n_real

    fake_frame = frame[~is_real]
    if len(fake_frame):
        groups = fake_frame.groupby(["dataset", "method"]).indices
        per_group = (1.0 - real_mass) / len(groups)
        positions = np.flatnonzero(~is_real)
        for _, local_indices in groups.items():
            weights[positions[local_indices]] = per_group / len(local_indices)

    return weights


def build_dataloaders_v2(
    data_dir: str | Path,
    manifest: str | Path,
    backbone_mean: tuple[float, ...],
    backbone_std: tuple[float, ...],
    batch_size: int = 32,
    num_workers: int = 6,
    image_size: int = 224,
    augment_strength: str = "heavy",
    exclude_methods: tuple[str, ...] = (),
    samples_per_epoch: int | None = None,
) -> dict[str, DataLoader]:
    train_transform = build_train_transform(
        image_size, backbone_mean, backbone_std, augment_strength
    )
    eval_transform = build_eval_transform(image_size, backbone_mean, backbone_std)

    train_set = ForgeryDataset(
        data_dir, manifest, "train", train_transform, exclude_methods=exclude_methods
    )
    val_set = ForgeryDataset(
        data_dir, manifest, "val", eval_transform, exclude_methods=exclude_methods
    )

    weights = balanced_weights(train_set.frame, train_set.sbi_prob)
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=samples_per_epoch or len(train_set),
        replacement=True,
    )

    common = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
        "prefetch_factor": 4 if num_workers > 0 else None,
    }
    return {
        "train": DataLoader(
            train_set, batch_size=batch_size, sampler=sampler, drop_last=True, **common
        ),
        "val": DataLoader(
            val_set, batch_size=batch_size * 2, shuffle=False, **common
        ),
    }
