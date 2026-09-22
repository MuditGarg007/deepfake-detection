"""Per-frame score distributions, so a weak method can be told from an inverted one.

``evaluate_v2.py`` answers "how well does it rank?"; this answers "where does it
put the scores?". The distinction decided the largest open question in the v2
work: SadTalker scored 0.4646 AUC with 2% recall at threshold 0.5, and an AUC
below 0.5 has exactly two readings —

*   the score distribution is **flat** (the model has no signal, and the number
    is 0.46 rather than 0.50 by sampling noise), or
*   it is **inverted** (the model has signal and is reading it backwards: these
    fakes score *lower* than genuine faces).

They call for opposite fixes. A flat method needs a cue the corpus does not
contain. An inverted method means a cue the corpus does contain is actively
firing the wrong way, which is a much stronger statement — and one that
generalises to every method sharing that generation mechanism.

So this script reports, per method and per generation family
(``protocol.FAMILY``): the score quantiles against the shared real pool, the
separation in units of the real pool's own spread, and a text histogram. It
writes ``scores.csv`` with one row per frame, so any further question can be
answered without another GPU pass.

Usage::

    python machine-learning/diagnose_v2.py \
        --checkpoint machine-learning/checkpoints/<run> --data data/processed_v2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

ML_DIR = Path(__file__).resolve().parent
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

import protocol  # noqa: E402
from augment import build_corruption, build_eval_transform  # noqa: E402
from dataset_v2 import ForgeryDataset  # noqa: E402
from evaluate_v2 import load_any, score, select_frames  # noqa: E402

BLOCKS = " ▁▂▃▄▅▆▇█"


def histogram(values: np.ndarray, bins: int = 20) -> str:
    """A one-line text histogram of scores over [0, 1]."""
    counts, _ = np.histogram(values, bins=bins, range=(0.0, 1.0))
    if not counts.max():
        return " " * bins
    scaled = counts / counts.max()
    return "".join(BLOCKS[min(int(v * (len(BLOCKS) - 1) + 0.999), len(BLOCKS) - 1)]
                   for v in scaled)


def summarise(frame: pd.DataFrame, probs: np.ndarray, key: str) -> pd.DataFrame:
    """Distribution statistics for every value of ``key`` against the reals."""
    is_real = (frame["label"] == "real").to_numpy()
    real_scores = probs[is_real]
    real_median = float(np.median(real_scores))
    # The real pool's own interquartile range is the natural yardstick: a shift
    # measured in raw probability means nothing when the whole distribution is
    # squashed into the bottom decile.
    real_iqr = float(np.subtract(*np.percentile(real_scores, [75, 25]))) or 1e-6

    rows = [{
        key: "REAL",
        "family": "real",
        "group": "-",
        "n": int(is_real.sum()),
        "median": real_median,
        "p10": float(np.percentile(real_scores, 10)),
        "p90": float(np.percentile(real_scores, 90)),
        "shift_iqr": 0.0,
        "auc": float("nan"),
        "recall@0.5": float((real_scores >= 0.5).mean()),
        "hist": histogram(real_scores),
    }]

    fake = frame[~is_real]
    for value, group in fake.groupby(key, sort=True):
        scores = probs[group.index.to_numpy()]
        truth = np.concatenate([np.zeros(len(real_scores)), np.ones(len(scores))])
        pooled = np.concatenate([real_scores, scores])
        rows.append({
            key: value,
            "family": protocol.family_for(group["method"].iloc[0])
            if key == "method" else value,
            "group": group["group"].iloc[0],
            "n": len(scores),
            "median": float(np.median(scores)),
            "p10": float(np.percentile(scores, 10)),
            "p90": float(np.percentile(scores, 90)),
            "shift_iqr": (float(np.median(scores)) - real_median) / real_iqr,
            "auc": float(roc_auc_score(truth, pooled)),
            "recall@0.5": float((scores >= 0.5).mean()),
            "hist": histogram(scores),
        })
    return pd.DataFrame(rows)


# AUC is the probability that a random fake outranks a random real, so it — not
# the median shift — is what says which side of chance a method falls on. With
# 800 fakes against 4,000 reals the standard error is around 0.01, so a method
# five points off 0.5 is off it for real. The bands below are deliberately
# wider than that: the question this answers is "what kind of failure is this",
# and that needs an effect size, not a p-value.
VERDICT_BANDS = (
    (0.90, "working"),   # usable
    (0.70, "weak"),      # ranks better than chance, misses a lot
    (0.55, "poor"),      # barely above chance
    (0.45, "flat"),      # indistinguishable from the real pool
)


def verdict(row: pd.Series) -> str:
    """What kind of failure this is, from the AUC band it falls in.

    ``flat`` and ``INVERTED`` are the two readings of a sub-chance AUC and they
    call for opposite fixes: a flat method needs a cue the corpus does not
    contain, while an inverted one means a cue the corpus *does* contain is
    firing backwards. ``shift_iqr`` is printed beside the verdict because the
    two cases also differ in the median — a flat method sits on top of the real
    pool, an inverted one sits below it.
    """
    if row["family"] == "real":
        return ""
    for floor, label in VERDICT_BANDS:
        if row["auc"] >= floor:
            return label
    return "INVERTED"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="data/processed_v2")
    parser.add_argument("--split", default="test")
    parser.add_argument("--corruption", default="clean")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--max-per-method", type=int, default=800)
    parser.add_argument("--max-real", type=int, default=4000)
    parser.add_argument("--heldout-all-splits", action="store_true",
                        help="score held-out fakes from every split, so the "
                             "flat image sets that hash entirely into train "
                             "are not skipped; reals stay inside --split")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    amp = args.amp and device.type == "cuda"
    model, config, mean, std, size = load_any(args.checkpoint, device)
    name = Path(args.checkpoint).name
    print(f"checkpoint {name}  (v{config.get('version', 1)}, "
          f"{config.get('backbone', config.get('model'))})  "
          f"corruption={args.corruption}", flush=True)

    manifest = pd.read_csv(Path(args.data) / "manifest.csv", low_memory=False)
    selection = select_frames(
        manifest, args.split, args.max_per_method, args.max_real,
        args.heldout_all_splits,
    )
    selection["family"] = selection["method"].map(protocol.family_for)
    print(f"{len(selection)} frames, "
          f"{selection['method'].nunique() - 1} manipulations, "
          f"{int((selection['label'] == 'real').sum())} real", flush=True)

    transform = (
        build_eval_transform(size, mean, std)
        if args.corruption == "clean"
        else build_corruption(args.corruption, size, mean, std)
    )
    loader = DataLoader(
        ForgeryDataset(args.data, selection, None, transform),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    probs = score(model, loader, device, amp)

    by_method = summarise(selection, probs, "method")
    by_method["verdict"] = by_method.apply(verdict, axis=1)
    by_method = by_method.sort_values(["group", "auc"], na_position="first")

    by_family = summarise(selection, probs, "family")
    by_family["verdict"] = by_family.apply(verdict, axis=1)

    columns = ["method", "family", "group", "n", "median", "p10", "p90",
               "shift_iqr", "auc", "recall@0.5", "verdict", "hist"]
    formatters = {c: (lambda v: f"{v:.3f}") for c in
                  ("median", "p10", "p90", "shift_iqr", "auc", "recall@0.5")}
    print("\nper-method score distribution "
          "(shift_iqr: fake median minus real median, in real-pool IQRs):")
    print(by_method[columns].to_string(index=False, formatters=formatters))

    print("\nper-family:")
    print(by_family[["family", "n", "median", "shift_iqr", "auc",
                     "recall@0.5", "verdict", "hist"]]
          .sort_values("auc", na_position="first")
          .to_string(index=False, formatters=formatters))

    inverted = by_method[by_method["verdict"] == "INVERTED"]["method"].tolist()
    flat = by_method[by_method["verdict"] == "flat"]["method"].tolist()
    print(f"\ninverted: {inverted or 'none'}")
    print(f"flat:     {flat or 'none'}")

    out_dir = Path(args.out) if args.out else ML_DIR / "runs" / f"diagnose_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    scores = selection[["path", "label", "dataset", "method", "family",
                        "group", "split"]].copy()
    scores["prob"] = probs
    scores.to_csv(out_dir / f"scores_{args.corruption}.csv", index=False)
    by_method.to_csv(out_dir / f"by_method_{args.corruption}.csv", index=False)
    by_family.to_csv(out_dir / f"by_family_{args.corruption}.csv", index=False)
    with open(out_dir / f"verdicts_{args.corruption}.json", "w",
              encoding="utf-8") as handle:
        json.dump({"checkpoint": str(args.checkpoint), "split": args.split,
                   "corruption": args.corruption,
                   "inverted": inverted, "flat": flat}, handle, indent=2)
    print(f"\nwrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
