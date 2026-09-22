"""Manifest-driven multi-source dataset for the v2 detector.

The v1 dataset had one corpus, one manipulation and a directory-per-split
layout. v2 mixes several corpora and ~30 manipulation methods, and the useful
questions are per-method ("does it catch InSwapper?"), so everything is driven
by one CSV with the provenance kept alongside the label:

``path,label,dataset,method,identity,split``

``label``     real / fake
``dataset``   ffpp, cdf, dfdcp, uadfv, df40
``method``    real, Deepfakes, Face2Face, inswap, simswap, sadtalker, ...
``identity``  source-video group, so a split never straddles one person
``split``     train / val / test / heldout

Two sampler details matter more here than in v1:

*   **Real/fake balance.** Every DF40 method reuses the same pool of real source
    videos, so naively pooling 25 manipulations gives ~25 fakes per real. Left
    alone the model just learns the prior. A ``WeightedRandomSampler`` equalises
    the two classes — and, because SBI relabels some sampled reals as fakes,
    the real mass is raised to compensate (``balanced_weights``).
*   **Method balance.** Within the fake class, methods with more frames would
    otherwise dominate. The same sampler equalises weight across methods, so a
    method contributes by existing rather than by frame count.
"""

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

cv2.setNumThreads(0)  # dataloader workers provide the parallelism


def read_rgb(path: Path) -> np.ndarray:
    """Load an image as an RGB uint8 array (cv2 is ~2x faster than PIL here)."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read image {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class ForgeryDataset(Dataset):
    """One split of the manifest.

    Yields ``(image_tensor, target, method_index)``; the method index lets the
    training loop report per-method validation AUC without a second pass.
    """

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

        # Self-blended pseudo-fakes are made here rather than pre-rendered to
        # disk: the point of SBI is that every draw is a different blend, and a
        # fixed set of them would be memorised like any other finite method.
        landmark = self._landmarks[index]
        if self.sbi_prob > 0 and target == 0.0 and landmark:
            # torch's RNG, not numpy's global one: DataLoader re-seeds torch per
            # worker but not numpy, so numpy's global stream would repeat
            # identically in every worker and collapse SBI's diversity.
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


# Above this SBI probability there is no real-mass allocation that balances the
# classes, because reals would generate more pseudo-fakes than reals. See
# ``balanced_weights``.
MAX_BALANCEABLE_SBI_PROB = 0.45


def real_mass_for(sbi_prob: float) -> float:
    """Sampling mass to give the real class so the *post-SBI* split is 50/50.

    SBI turns a sampled real frame into a fake one with probability ``p``, so a
    naive 50/50 sampler actually trains on ``(1-p)/2`` real and ``(1+p)/2``
    fake. At ``p = 0.5`` that is 25/75, and the model learns a fake-leaning
    prior — measured as a 0.459 false-positive rate on real faces at threshold
    0.5, which is useless in production even though AUC looks fine.

    Solving ``R(1-p) = Rp + (1-R)`` for the real mass ``R`` gives::

        R = 1 / (2 * (1 - p))

    which exceeds 1 once ``p >= 0.5`` — at that point every real frame would
    have to be sampled and there would be no budget left for real
    manipulations, so the caller must keep ``p`` below
    ``MAX_BALANCEABLE_SBI_PROB``.
    """
    if sbi_prob <= 0:
        return 0.5
    if sbi_prob >= MAX_BALANCEABLE_SBI_PROB:
        raise ValueError(
            f"sbi_prob={sbi_prob} cannot be class-balanced; keep it below "
            f"{MAX_BALANCEABLE_SBI_PROB}"
        )
    return 1.0 / (2.0 * (1.0 - sbi_prob))


def balanced_weights(frame: pd.DataFrame, sbi_prob: float = 0.0) -> np.ndarray:
    """Per-sample weights equalising real/fake and, inside fake, method.

    Reals are treated as one group so a corpus with many real frames does not
    outvote the others; fakes are split per ``(dataset, method)`` so each
    manipulation carries the same total mass.

    ``sbi_prob`` must match the value handed to ``ForgeryDataset``, so the real
    mass can be raised to compensate for the reals that SBI will relabel as
    fakes. Only reals that actually carry landmarks can be converted, so the
    effective conversion rate is scaled by that fraction.
    """
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
        # ``groups`` indexes into fake_frame; map back to manifest positions.
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
    """Train loader (balanced + augmented) and val loader (plain)."""
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
