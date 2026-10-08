from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

ML_DIR = Path(__file__).resolve().parent
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

import protocol  # noqa: E402
from augment import CORRUPTIONS, build_corruption, build_eval_transform  # noqa: E402
from dataset_v2 import ForgeryDataset  # noqa: E402


def load_any(checkpoint_dir: str | Path, device: torch.device):
    directory = Path(checkpoint_dir)
    with open(directory / "config.json", encoding="utf-8") as handle:
        config = json.load(handle)
    state = torch.load(directory / "best.pth", map_location="cpu")

    if config.get("version") == 2:
        from models_v2 import Detector, normalization_for

        model = Detector(
            config["backbone"], pretrained=False,
            tune=config.get("tune", "ln"), head=config.get("head", "hypersphere"),
        )
        model.load_state_dict(state)
        mean, std = tuple(config["mean"]), tuple(config["std"])
        size = config.get("image_size", 224)
    else:
        from models import get_model

        model = get_model(config["model"], pretrained=False)
        model.load_state_dict(state)
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        size = 224

    model.to(device).eval()
    return model, config, mean, std, size


@torch.no_grad()
def score(model, loader, device, amp: bool) -> np.ndarray:
    probabilities = []
    for images, _, _ in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            logits = model(images)
        logits = logits.squeeze(1) if logits.ndim == 2 else logits
        probabilities.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(probabilities)


def auc_against_reals(probs: np.ndarray, frame: pd.DataFrame, key: str) -> pd.DataFrame:
    is_real = (frame["label"] == "real").to_numpy()
    real_scores = probs[is_real]
    if not len(real_scores):
        raise ValueError("no real frames in this selection")

    rows = []
    fake = frame[~is_real]
    for value, group in fake.groupby(key, sort=True):
        fake_scores = probs[group.index.to_numpy()]
        scores = np.concatenate([real_scores, fake_scores])
        truth = np.concatenate([np.zeros(len(real_scores)), np.ones(len(fake_scores))])
        rows.append(
            {
                key: value,
                "dataset": group["dataset"].iloc[0],
                "group": group["group"].iloc[0],
                "n_fake": len(fake_scores),
                "auc": roc_auc_score(truth, scores),
                "ap": average_precision_score(truth, scores),
                "recall@0.5": float((fake_scores >= 0.5).mean()),
            }
        )
    table = pd.DataFrame(rows).sort_values(["group", "auc"])
    table.attrs["real_fpr@0.5"] = float((real_scores >= 0.5).mean())
    table.attrs["n_real"] = int(len(real_scores))
    return table


def select_frames(manifest: pd.DataFrame, split: str, max_per_method: int,
                  max_real: int, heldout_all_splits: bool = False,
                  seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    def cap(group: pd.DataFrame, limit: int) -> pd.DataFrame:
        if len(group) <= limit:
            return group
        return group.iloc[rng.choice(len(group), limit, replace=False)]

    in_split = manifest[manifest["split"] == split]
    reals = cap(in_split[in_split["label"] == "real"], max_real)

    fake_pool = in_split[in_split["label"] == "fake"]
    if heldout_all_splits:
        heldout = manifest[
            (manifest["label"] == "fake") & (manifest["group"] == "heldout")
        ]
        fake_pool = pd.concat([fake_pool, heldout]).drop_duplicates(subset="path")

    fakes = pd.concat(
        [cap(group, max_per_method) for _, group in fake_pool.groupby("method")]
    )
    return pd.concat([reals, fakes]).reset_index(drop=True)


def video_level(probs: np.ndarray, frame: pd.DataFrame) -> dict[str, float]:
    work = frame.copy()
    work["prob"] = probs
    work["video"] = work["path"].map(lambda p: str(Path(p).parent))
    per_video = work.groupby("video").agg(
        prob=("prob", "mean"), label=("label", "first")
    )
    truth = (per_video["label"] == "fake").to_numpy().astype(float)
    if len(np.unique(truth)) < 2:
        return {"video_auc": float("nan"), "n_videos": len(per_video)}
    return {
        "video_auc": float(roc_auc_score(truth, per_video["prob"].to_numpy())),
        "n_videos": int(len(per_video)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="data/processed_v2")
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--max-per-method", type=int, default=2000,
                        help="cap frames per method to keep the sweep quick")
    parser.add_argument("--max-real", type=int, default=6000)
    parser.add_argument("--corruptions", nargs="*", default=None,
                        help="default: the full sweep; pass 'clean' to skip it")
    parser.add_argument("--heldout-all-splits", action="store_true",
                        help="score held-out fakes from every split, so the "
                             "flat image sets that hash entirely into train "
                             "are not skipped; reals stay inside --split")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--out", default=None, help="directory for the JSON/CSV report")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    amp = args.amp and device.type == "cuda"
    model, config, mean, std, size = load_any(args.checkpoint, device)
    name = Path(args.checkpoint).name
    print(f"checkpoint {name}  (v{config.get('version', 1)}, "
          f"{config.get('backbone', config.get('model'))})", flush=True)

    manifest = pd.read_csv(Path(args.data) / "manifest.csv", low_memory=False)
    selection = select_frames(
        manifest, args.split, args.max_per_method, args.max_real,
        args.heldout_all_splits,
    )
    n_real = int((selection["label"] == "real").sum())
    print(f"split={args.split}: {len(selection)} frames, "
          f"{selection['method'].nunique() - 1} manipulations, "
          f"{n_real} real", flush=True)

    corruptions = args.corruptions or list(CORRUPTIONS)
    report: dict[str, dict] = {}
    tables: list[pd.DataFrame] = []

    for corruption in corruptions:
        transform = (
            build_eval_transform(size, mean, std)
            if corruption == "clean"
            else build_corruption(corruption, size, mean, std)
        )
        dataset = ForgeryDataset(args.data, selection, None, transform)
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=device.type == "cuda",
        )
        probs = score(model, loader, device, amp)
        table = auc_against_reals(probs, selection, "method")
        table.insert(0, "corruption", corruption)
        tables.append(table)

        by_group = table.groupby("group")["auc"].mean().to_dict()
        overall = {
            "macro_auc": float(table["auc"].mean()),
            "heldout_macro_auc": float(by_group.get("heldout", float("nan"))),
            "heldin_macro_auc": float(by_group.get("heldin", float("nan"))),
            "real_fpr@0.5": table.attrs["real_fpr@0.5"],
            **video_level(probs, selection),
        }
        report[corruption] = {"overall": overall, "per_group": by_group}
        print(
            f"  {corruption:<15} macro {overall['macro_auc']:.4f} | "
            f"held-out {overall['heldout_macro_auc']:.4f} | "
            f"video {overall['video_auc']:.4f} | "
            f"real FPR@0.5 {overall['real_fpr@0.5']:.3f}",
            flush=True,
        )

    full = pd.concat(tables, ignore_index=True)
    clean = full[full["corruption"] == "clean"].sort_values(["group", "auc"])
    print("\nper-method AUC (clean):")
    print(clean[["method", "dataset", "group", "n_fake", "auc", "recall@0.5"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    out_dir = Path(args.out) if args.out else ML_DIR / "runs" / f"eval_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    full.to_csv(out_dir / "per_method.csv", index=False)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(
            {"checkpoint": str(args.checkpoint), "split": args.split,
             "config": config, "report": report},
            handle, indent=2,
        )
    print(f"\nwrote {out_dir}/per_method.csv and summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
