"""T7 — test-set metrics and plots for a trained checkpoint.

Usage::

    python machine-learning/evaluate.py \
        --checkpoint machine-learning/checkpoints/efficientnet_b0_<ts> \
        --data data/processed

Writes into ``machine-learning/runs/<run_name>/``:
    metrics.json, roc_curve.png, confusion_matrix.png, thresholds.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm  # noqa: E402

ML_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ML_DIR))

from dataset import FaceDataset  # noqa: E402
from inference import load_model  # noqa: E402


@torch.no_grad()
def collect_predictions(model, loader, device, amp: bool):
    """Return ``(probabilities, targets)`` over the whole loader."""
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for images, target in tqdm(loader, desc="test"):
        images = images.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            logits = model(images).squeeze(1)
        probabilities.append(torch.sigmoid(logits.float()).cpu().numpy())
        targets.append(target.numpy())
    return np.concatenate(probabilities), np.concatenate(targets)


@torch.no_grad()
def measure_latency(model, device, amp: bool, runs: int = 100) -> float:
    """Mean single-image forward time in milliseconds."""
    sample = torch.randn(1, 3, 224, 224, device=device)
    if device.type == "cuda":
        sample = sample.to(memory_format=torch.channels_last)
    for _ in range(10):  # warm up kernels / autotuning
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            model(sample)
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(runs):
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            model(sample)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / runs * 1000


def plot_roc(truth, probs, auc: float, path: Path, title: str) -> None:
    fpr, tpr, _ = roc_curve(truth, probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="chance")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"ROC — {title}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_confusion(matrix, path: Path, title: str) -> None:
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["real", "fake"],
        yticklabels=["real", "fake"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion matrix — {title}")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def write_thresholds(truth, probs, path: Path) -> None:
    """Precision/recall/F1/accuracy per threshold — feeds Phase 2's risk cutoffs."""
    precision, recall, thresholds = precision_recall_curve(truth, probs)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["threshold", "precision", "recall", "f1", "accuracy"])
        for i, threshold in enumerate(thresholds):
            p, r = precision[i], recall[i]
            f1 = 2 * p * r / (p + r) if (p + r) else 0.0
            accuracy = float(((probs >= threshold).astype(int) == truth).mean())
            writer.writerow(
                [
                    round(float(threshold), 6),
                    round(float(p), 6),
                    round(float(r), 6),
                    round(float(f1), 6),
                    round(accuracy, 6),
                ]
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="checkpoint directory")
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--out", default=None, help="output dir (default runs/<run_name>)")
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint)
    with open(ckpt_dir / "config.json", encoding="utf-8") as handle:
        config = json.load(handle)

    device = torch.device(args.device)
    amp = args.amp and device.type == "cuda"
    model = load_model(ckpt_dir, device=device)

    dataset = FaceDataset(args.data, args.split, train=False)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    print(f"{args.split}: {dataset.class_counts()}")

    probs, truth = collect_predictions(model, loader, device, amp)
    predictions = (probs >= args.threshold).astype(int)

    metrics = {
        "model": config["model"],
        "run_name": config.get("run_name", ckpt_dir.name),
        "checkpoint": str(ckpt_dir),
        "split": args.split,
        "threshold": args.threshold,
        "num_samples": int(len(truth)),
        "class_counts": dataset.class_counts(),
        "accuracy": float(accuracy_score(truth, predictions)),
        "precision": float(precision_score(truth, predictions, zero_division=0)),
        "recall": float(recall_score(truth, predictions, zero_division=0)),
        "f1": float(f1_score(truth, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(truth, probs)),
        "inference_ms_per_image": round(measure_latency(model, device, amp), 3),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
    }
    matrix = confusion_matrix(truth, predictions)
    tn, fp, fn, tp = matrix.ravel()
    metrics["confusion_matrix"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}

    out_dir = Path(args.out) if args.out else ML_DIR / "runs" / metrics["run_name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    plot_roc(truth, probs, metrics["roc_auc"], out_dir / "roc_curve.png", config["model"])
    plot_confusion(matrix, out_dir / "confusion_matrix.png", config["model"])
    write_thresholds(truth, probs, out_dir / "thresholds.csv")

    print(json.dumps({k: v for k, v in metrics.items() if k != "class_counts"}, indent=2))
    print(f"\nartifacts -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
