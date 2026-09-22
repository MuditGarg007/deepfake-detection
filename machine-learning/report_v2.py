"""Collect the sweep's evaluations into one comparison, and score ensembles.

Two modes:

``--compare`` (default)
    Reads every ``runs/eval_*/summary.json`` and ``per_method.csv`` and writes
    ``runs/v2_comparison.md``: one row per run, ranked by held-out macro AUC —
    the number the whole exercise is about — with the in-domain, corruption and
    cross-corpus columns beside it so a run that wins by overfitting is visible.

``--ensemble CKPT [CKPT ...]``
    Scores several checkpoints on the same frames and reports the mean of their
    probabilities alongside each member. Architecturally diverse members
    (a CLIP ViT and a DINOv3 ViT make different mistakes) usually beat the best
    single model on unseen methods, which is the finding this is here to check
    rather than assume.
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

from augment import build_corruption, build_eval_transform  # noqa: E402
from dataset_v2 import ForgeryDataset  # noqa: E402
from evaluate_v2 import (  # noqa: E402
    auc_against_reals, load_any, score, select_frames, video_level,
)

# Corruptions summarised in the comparison table; the full sweep stays in each
# run's per_method.csv.
HEADLINE_CORRUPTIONS = ("clean", "jpeg_q40", "jpeg_q10", "downscale_0.25")


def collect(runs_dir: Path) -> pd.DataFrame:
    rows = []
    for summary_path in sorted(runs_dir.glob("eval_*/summary.json")):
        with open(summary_path, encoding="utf-8") as handle:
            payload = json.load(handle)
        report = payload["report"]
        config = payload.get("config", {})
        clean = report.get("clean", {}).get("overall", {})
        row = {
            "run": summary_path.parent.name.removeprefix("eval_"),
            "backbone": config.get("backbone", config.get("model", "?")),
            "tune": config.get("tune", "full"),
            "heldout_auc": clean.get("heldout_macro_auc", float("nan")),
            "heldin_auc": clean.get("heldin_macro_auc", float("nan")),
            "video_auc": clean.get("video_auc", float("nan")),
            "real_fpr": clean.get("real_fpr@0.5", float("nan")),
        }
        for corruption in HEADLINE_CORRUPTIONS[1:]:
            row[corruption] = (
                report.get(corruption, {}).get("overall", {}).get(
                    "heldout_macro_auc", float("nan")
                )
            )
        # Worst held-out method is the number that decides whether the detector
        # is trustworthy: an average hides a manipulation it cannot see at all.
        per_method_path = summary_path.parent / "per_method.csv"
        if per_method_path.is_file():
            table = pd.read_csv(per_method_path)
            table = table[(table["corruption"] == "clean") & (table["group"] == "heldout")]
            if len(table):
                row["worst_heldout"] = float(table["auc"].min())
                row["worst_method"] = table.loc[table["auc"].idxmin(), "method"]
        rows.append(row)
    if not rows:
        raise SystemExit(f"no eval_*/summary.json found under {runs_dir}")
    return pd.DataFrame(rows).sort_values("heldout_auc", ascending=False)


def write_comparison(runs_dir: Path) -> Path:
    table = collect(runs_dir)
    out = runs_dir / "v2_comparison.md"
    columns = [
        "run", "backbone", "tune", "heldout_auc", "worst_heldout", "worst_method",
        "heldin_auc", "video_auc", "jpeg_q40", "jpeg_q10", "downscale_0.25",
        "real_fpr",
    ]
    columns = [c for c in columns if c in table.columns]
    body = table[columns].to_markdown(index=False, floatfmt=".4f")
    with open(out, "w", encoding="utf-8") as handle:
        handle.write("# v2 sweep comparison\n\n")
        handle.write(
            "Ranked by `heldout_auc`: the mean per-method AUC over manipulations "
            "and corpora that were excluded from training. `heldin_auc` is the "
            "same quantity over the manipulations the model trained on — a large "
            "gap between the two is the overfitting this work exists to remove. "
            "`worst_heldout` is the single weakest held-out manipulation.\n\n"
        )
        handle.write(body + "\n")
    print(body)
    print(f"\nwrote {out}")
    return out


def run_ensemble(checkpoints: list[str], data: str, split: str, device: torch.device,
                 amp: bool, max_per_method: int, max_real: int,
                 corruptions: list[str], batch_size: int, num_workers: int,
                 heldout_all_splits: bool = False) -> None:
    manifest = pd.read_csv(Path(data) / "manifest.csv", low_memory=False)
    selection = select_frames(manifest, split, max_per_method, max_real,
                              heldout_all_splits)
    print(f"{len(selection)} frames, {selection['method'].nunique() - 1} manipulations")

    models = {}
    for checkpoint in checkpoints:
        name = Path(checkpoint).name
        models[name] = load_any(checkpoint, device)
        print(f"loaded {name}")

    for corruption in corruptions:
        member_probs = {}
        for name, (model, _config, mean, std, size) in models.items():
            transform = (
                build_eval_transform(size, mean, std)
                if corruption == "clean"
                else build_corruption(corruption, size, mean, std)
            )
            loader = DataLoader(
                ForgeryDataset(data, selection, None, transform),
                batch_size=batch_size, shuffle=False, num_workers=num_workers,
                pin_memory=device.type == "cuda",
            )
            member_probs[name] = score(model, loader, device, amp)

        # Rank-average rather than a plain mean: the members are calibrated
        # differently, so averaging raw probabilities lets the most confident
        # model dominate regardless of whether it is the most correct one.
        stacked = np.stack(list(member_probs.values()))
        ranks = np.stack([pd.Series(p).rank(pct=True).to_numpy() for p in stacked])
        member_probs["ENSEMBLE(mean)"] = stacked.mean(axis=0)
        member_probs["ENSEMBLE(rank)"] = ranks.mean(axis=0)

        print(f"\n--- {corruption}")
        for name, probs in member_probs.items():
            table = auc_against_reals(probs, selection, "method")
            heldout = table[table["group"] == "heldout"]["auc"].mean()
            video = video_level(probs, selection)["video_auc"]
            print(f"  {name:<44} held-out macro {heldout:.4f}  "
                  f"worst {table[table['group'] == 'heldout']['auc'].min():.4f}  "
                  f"video {video:.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default=str(ML_DIR / "runs"))
    parser.add_argument("--ensemble", nargs="*", default=None,
                        help="checkpoint directories to score together")
    parser.add_argument("--data", default="data/processed_v2")
    parser.add_argument("--split", default="test")
    parser.add_argument("--corruptions", nargs="*", default=["clean", "jpeg_q40"])
    parser.add_argument("--max-per-method", type=int, default=1200)
    parser.add_argument("--max-real", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--heldout-all-splits", action="store_true",
                        help="score held-out fakes from every split; matches "
                             "the flag evaluate_v2.py uses")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.ensemble:
        device = torch.device(args.device)
        run_ensemble(
            args.ensemble, args.data, args.split, device,
            args.amp and device.type == "cuda", args.max_per_method,
            args.max_real, args.corruptions, args.batch_size, args.num_workers,
            args.heldout_all_splits,
        )
        return 0

    write_comparison(Path(args.runs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
