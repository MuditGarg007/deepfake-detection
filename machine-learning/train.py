"""T6 — fine-tune one backbone on the face-crop dataset.

Usage::

    python machine-learning/train.py --model efficientnet_b0 --data data/processed \
        --epochs 15 --batch-size 64 --lr 1e-4 --amp --patience 5 --seed 42

Writes:
    machine-learning/checkpoints/<model>_<timestamp>/best.pth
    machine-learning/checkpoints/<model>_<timestamp>/config.json
    machine-learning/runs/<model>_<timestamp>/log.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

ML_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ML_DIR))

from dataset import build_dataloaders  # noqa: E402
from models import MODEL_NAMES, get_model  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_split(model, loader, criterion, device, amp: bool) -> dict[str, float]:
    """Loss / accuracy / ROC-AUC over a whole loader."""
    model.eval()
    total_loss = 0.0
    seen = 0
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    for images, target in loader:
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            logits = model(images).squeeze(1)
            loss = criterion(logits.float(), target)
        total_loss += loss.item() * images.size(0)
        seen += images.size(0)
        probabilities.append(torch.sigmoid(logits.float()).cpu().numpy())
        targets.append(target.cpu().numpy())

    probs = np.concatenate(probabilities)
    truth = np.concatenate(targets)
    accuracy = float(((probs >= 0.5).astype(float) == truth).mean())
    # A split with a single class present has no defined AUC.
    auc = float(roc_auc_score(truth, probs)) if len(np.unique(truth)) > 1 else float("nan")
    return {"loss": total_loss / seen, "acc": accuracy, "auc": auc}


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, amp, clip):
    model.train()
    total_loss = 0.0
    correct = 0
    seen = 0

    for images, target in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            logits = model(images).squeeze(1)
            loss = criterion(logits.float(), target)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        correct += int(((torch.sigmoid(logits.float()) >= 0.5).float() == target).sum())
        seen += images.size(0)

    return {"loss": total_loss / seen, "acc": correct / seen}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODEL_NAMES), required=True)
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=5, help="early-stop patience")
    parser.add_argument("--clip", type=float, default=1.0, help="grad-norm clip")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true", help="mixed-precision training")
    parser.add_argument("--out", default=str(ML_DIR), help="root for checkpoints/ + runs/")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    amp = args.amp and device.type == "cuda"

    loaders = build_dataloaders(
        args.data, args.batch_size, args.num_workers, splits=("train", "val")
    )
    counts = {
        split: loaders[split].dataset.class_counts() for split in ("train", "val")
    }
    print(f"train: {counts['train']}   val: {counts['val']}")

    model = get_model(args.model).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    # Rebalance the loss if the training split is skewed.
    train_counts = counts["train"]
    pos_weight = torch.tensor(
        [train_counts.get("real", 1) / max(train_counts.get("fake", 1), 1)],
        device=device,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.model}_{stamp}"
    out_root = Path(args.out)
    ckpt_dir = out_root / "checkpoints" / run_name
    run_dir = out_root / "runs" / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "log.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(
            ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_auc", "seconds"]
        )

    best_auc = -1.0
    best_epoch = -1
    best_val: dict[str, float] = {}
    epochs_without_gain = 0
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_metrics = train_one_epoch(
            model, loaders["train"], criterion, optimizer, scaler, device, amp, args.clip
        )
        val_metrics = evaluate_split(model, loaders["val"], criterion, device, amp)
        elapsed = time.time() - epoch_start

        print(
            f"epoch {epoch:>2}/{args.epochs} "
            f"train loss {train_metrics['loss']:.4f} acc {train_metrics['acc']:.4f} | "
            f"val loss {val_metrics['loss']:.4f} acc {val_metrics['acc']:.4f} "
            f"auc {val_metrics['auc']:.4f} | {elapsed:.0f}s",
            flush=True,
        )
        with open(log_path, "a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(
                [
                    epoch,
                    round(train_metrics["loss"], 6),
                    round(train_metrics["acc"], 6),
                    round(val_metrics["loss"], 6),
                    round(val_metrics["acc"], 6),
                    round(val_metrics["auc"], 6),
                    round(elapsed, 1),
                ]
            )

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            best_epoch = epoch
            best_val = val_metrics
            epochs_without_gain = 0
            torch.save(model.state_dict(), ckpt_dir / "best.pth")
            print(f"  new best val AUC {best_auc:.4f} -> {ckpt_dir / 'best.pth'}")
        else:
            epochs_without_gain += 1
            if epochs_without_gain >= args.patience:
                print(f"early stopping: no val AUC gain for {args.patience} epochs")
                break

    config = {
        "model": args.model,
        "timm_name": MODEL_NAMES[args.model],
        "run_name": run_name,
        "args": vars(args),
        "image_size": 224,
        "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "class_mapping": {"real": 0, "fake": 1},
        "split_counts": counts,
        "pos_weight": float(pos_weight.item()),
        "best_epoch": best_epoch,
        "best_val": best_val,
        "train_seconds": round(time.time() - started, 1),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "amp": amp,
    }
    with open(ckpt_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    print(f"\nbest epoch {best_epoch} val AUC {best_auc:.4f}")
    print(f"checkpoint -> {ckpt_dir}")
    print(f"log        -> {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
