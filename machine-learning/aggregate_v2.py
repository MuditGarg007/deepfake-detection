"""Which way of turning frame scores into a video verdict is actually best?

`video_processor.aggregate` takes the plain mean of a video's frame scores, and
`evaluate_v2.py` matches it so the reported video AUC means what the product
does. Neither choice was ever measured, and the obvious alternatives pull in
opposite directions: a trimmed mean is robust to a few bad frames, while a
top-k mean is the right shape for a manipulation that only touches part of a
clip.

This reads the per-frame scores `diagnose_v2.py` already wrote — no GPU — and
compares aggregators on the two things that matter separately:

*   **Ranking**, as pooled AUC and as held-out macro AUC. These are
    threshold-free and can disagree with each other.
*   **Decisions**, as recall at a *matched* false-positive rate. Comparing
    recall at a shared 0.5 cutoff instead would reward whichever aggregator
    pushes scores up, which is not the same as rewarding a better one.

Usage::

    python machine-learning/aggregate_v2.py \
        --scores machine-learning/runs/diagnose_<run>/scores_clean.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ML_DIR = Path(__file__).resolve().parent

AGGREGATORS = ("mean", "median", "trimmed20", "q90", "top50", "top25", "max")


def aggregate(scores: np.ndarray, how: str) -> float:
    values = np.sort(scores)
    if how == "mean":
        return float(values.mean())
    if how == "median":
        return float(np.median(values))
    if how == "trimmed20":
        k = max(1, int(len(values) * 0.2))
        trimmed = values[k:len(values) - k]
        return float(trimmed.mean() if len(trimmed) else values.mean())
    if how == "q90":
        return float(np.quantile(values, 0.9))
    if how.startswith("top"):
        share = int(how.removeprefix("top")) / 100
        k = max(1, int(round(len(values) * share)))
        return float(values[-k:].mean())
    if how == "max":
        return float(values[-1])
    raise ValueError(f"unknown aggregator '{how}'")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True,
                        help="scores_<corruption>.csv written by diagnose_v2.py")
    parser.add_argument("--min-frames", type=int, default=3,
                        help="folders with fewer frames are not videos")
    parser.add_argument("--fpr", type=float, default=0.05,
                        help="false-positive rate the recall comparison is "
                             "matched at")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    frame = pd.read_csv(args.scores)
    frame["video"] = frame["path"].map(lambda p: str(Path(p).parent))
    counts = frame.groupby("video")["prob"].size()
    frame = frame[frame["video"].map(counts) >= args.min_frames]

    per_video = frame.groupby("video").agg(
        label=("label", "first"), group=("group", "first"),
        method=("method", "first"),
    )
    for how in AGGREGATORS:
        per_video[how] = frame.groupby("video")["prob"].apply(
            lambda g, h=how: aggregate(g.to_numpy(), h)
        )

    real = per_video[per_video["label"] == "real"]
    fake = per_video[per_video["label"] == "fake"]
    heldout = fake[fake["group"] == "heldout"]
    print(f"{len(real)} real videos, {len(fake)} fake "
          f"({len(heldout)} held-out), matched FPR {args.fpr:.0%}\n")

    truth = (per_video["label"] == "fake").astype(int)
    rows = []
    print(f"{'aggregator':<12}{'pooled AUC':>12}{'heldout macro':>15}"
          f"{'FPR@0.5':>10}{'threshold':>11}{'recall':>9}{'recall(ho)':>12}")
    for how in AGGREGATORS:
        per_method = [
            roc_auc_score(
                np.r_[np.zeros(len(real)), np.ones(len(group))],
                np.r_[real[how], group[how]],
            )
            for _, group in heldout.groupby("method")
        ]
        threshold = float(np.quantile(real[how], 1 - args.fpr))
        row = {
            "aggregator": how,
            "pooled_auc": float(roc_auc_score(truth, per_video[how])),
            "heldout_macro_auc": float(np.mean(per_method)),
            "real_fpr@0.5": float((real[how] >= 0.5).mean()),
            "threshold": threshold,
            "recall": float((fake[how] >= threshold).mean()),
            "heldout_recall": float((heldout[how] >= threshold).mean()),
        }
        rows.append(row)
        print(f"{how:<12}{row['pooled_auc']:>12.4f}"
              f"{row['heldout_macro_auc']:>15.4f}{row['real_fpr@0.5']:>10.4f}"
              f"{threshold:>11.4f}{row['recall']:>9.4f}"
              f"{row['heldout_recall']:>12.4f}")

    table = pd.DataFrame(rows)
    best_rank = table.loc[table["heldout_macro_auc"].idxmax(), "aggregator"]
    best_decide = table.loc[table["heldout_recall"].idxmax(), "aggregator"]
    print(f"\nbest held-out macro AUC: {best_rank}")
    print(f"best held-out recall at matched FPR: {best_decide}")
    if best_rank != best_decide:
        print("  — they disagree, and the second is the one the product does: "
              "AUC rewards an aggregator that spreads the scores out, while a "
              "matched-FPR recall asks what it buys at a fixed budget")

    out = Path(args.out) if args.out else ML_DIR / "runs" / "video_aggregation.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
