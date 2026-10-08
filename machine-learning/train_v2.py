from __future__ import annotations

import argparse
import csv
import json
import math
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
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

import protocol  # noqa: E402
from augment import build_eval_transform, build_train_transform  # noqa: E402
from dataset_v2 import ForgeryDataset, balanced_weights  # noqa: E402
from models_v2 import BACKBONES, Detector, normalization_for  # noqa: E402
from torch.utils.data import DataLoader, WeightedRandomSampler  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def macro_auc(probs: np.ndarray, targets: np.ndarray, methods: np.ndarray,
              method_names: list[str]) -> tuple[float, dict[str, float]]:
    real_mask = targets == 0.0
    real_scores = probs[real_mask]
    per_method: dict[str, float] = {}
    for index, name in enumerate(method_names):
        if name == "real":
            continue
        fake_mask = (targets == 1.0) & (methods == index)
        if not fake_mask.any() or not real_mask.any():
            continue
        scores = np.concatenate([real_scores, probs[fake_mask]])
        truth = np.concatenate([np.zeros(real_mask.sum()), np.ones(fake_mask.sum())])
        per_method[name] = float(roc_auc_score(truth, scores))
    mean = float(np.mean(list(per_method.values()))) if per_method else float("nan")
    return mean, per_method


@torch.no_grad()
def run_eval(model, loader, device, amp: bool) -> dict:
    model.eval()
    probs, targets, methods = [], [], []
    for images, target, method in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            logits = model(images).squeeze(1)
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
        targets.append(target.numpy())
        methods.append(method.numpy())
    probs = np.concatenate(probs)
    targets = np.concatenate(targets)
    methods = np.concatenate(methods)
    names = loader.dataset.method_names
    mean, per_method = macro_auc(probs, targets, methods, names)
    pooled = (
        float(roc_auc_score(targets, probs))
        if len(np.unique(targets)) > 1
        else float("nan")
    )
    accuracy = float(((probs >= 0.5).astype(float) == targets).mean())
    return {
        "macro_auc": mean,
        "pooled_auc": pooled,
        "acc": accuracy,
        "per_method": per_method,
    }


def latent_augment(features: torch.Tensor, targets: torch.Tensor,
                   noise: float, mixup: float):
    if noise > 0:
        features = features + noise * torch.randn_like(features) * features.norm(
            dim=-1, keepdim=True
        ) / math.sqrt(features.shape[-1])
    if mixup > 0:
        permutation = torch.randperm(features.shape[0], device=features.device)
        same_class = (targets == targets[permutation]).float().unsqueeze(1)
        lam = torch.empty(features.shape[0], 1, device=features.device).uniform_(
            1.0 - mixup, 1.0
        )
        lam = 1.0 - (1.0 - lam) * same_class
        features = lam * features + (1.0 - lam) * features[permutation]
    return features


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler,
                    device, amp, clip, noise, mixup) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    seen = 0
    progress = tqdm(
        loader, desc="train", leave=False,
        disable=not sys.stderr.isatty(), mininterval=10.0,
    )
    for images, target, _ in progress:
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            features = model.features(images)
            features = latent_augment(features.float(), target, noise, mixup)
            logits = model.head(features).squeeze(1)
            loss = criterion(logits.float(), target)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.trainable_parameters(), clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item() * images.size(0)
        correct += int(((torch.sigmoid(logits.float()) >= 0.5).float() == target).sum())
        seen += images.size(0)
    return {"loss": total_loss / seen, "acc": correct / seen}


def build_loaders(args, mean, std):
    manifest = Path(args.data) / "manifest.csv"
    import pandas as pd

    frame = pd.read_csv(manifest)

    train_frame = frame[(frame["split"] == "train") & (frame["group"] == "heldin")]
    val_frame = frame[(frame["split"] == "val") & (frame["group"] == "heldin")]
    unseen_frame = frame[
        (frame["split"] == "val")
        & ((frame["group"] == "valunseen") | (frame["label"] == "real"))
    ]

    train_transform = build_train_transform(args.image_size, mean, std, args.augment)
    eval_transform = build_eval_transform(args.image_size, mean, std)

    train_set = ForgeryDataset(
        args.data, train_frame, None, train_transform, sbi_prob=args.sbi_prob
    )
    val_set = ForgeryDataset(args.data, val_frame, None, eval_transform)
    unseen_set = ForgeryDataset(args.data, unseen_frame, None, eval_transform)

    sampler = WeightedRandomSampler(
        torch.as_tensor(
            balanced_weights(train_set.frame, args.sbi_prob), dtype=torch.double
        ),
        num_samples=args.samples_per_epoch or len(train_set),
        replacement=True,
    )
    common = {
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": args.num_workers > 0,
        "prefetch_factor": 4 if args.num_workers > 0 else None,
    }
    eval_batch = max(args.batch_size, 32)
    return {
        "train": DataLoader(
            train_set, batch_size=args.batch_size, sampler=sampler,
            drop_last=True, **common,
        ),
        "val": DataLoader(val_set, batch_size=eval_batch, shuffle=False, **common),
        "val_unseen": DataLoader(
            unseen_set, batch_size=eval_batch, shuffle=False, **common
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", choices=sorted(BACKBONES), default="clip_vit_l14")
    parser.add_argument("--tune", choices=("full", "ln", "head"), default="ln")
    parser.add_argument("--head", choices=("hypersphere", "mlp"), default="hypersphere")
    parser.add_argument("--data", default="data/processed_v2")
    parser.add_argument(
        "--augment", choices=("none", "light", "heavy", "heavy_wide"), default="heavy"
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-frac", type=float, default=0.05)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--sbi-prob", type=float, default=0.25,
                        help="chance a real training frame becomes a self-blended "
                             "pseudo-fake; 0 disables SBI")
    parser.add_argument("--feature-noise", type=float, default=0.1)
    parser.add_argument("--feature-mixup", type=float, default=0.2)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--samples-per-epoch", type=int, default=60000)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad-checkpointing", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default=str(ML_DIR))
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    amp = args.amp and device.type == "cuda"

    mean, std = normalization_for(args.backbone)
    loaders = build_loaders(args, mean, std)
    for split, loader in loaders.items():
        frame = loader.dataset.frame
        counts = frame["label"].value_counts().to_dict()
        print(f"{split:<10} {len(frame):>7} frames  {counts}  "
              f"{len(frame['method'].unique())} methods", flush=True)

    model = Detector(
        args.backbone,
        pretrained=True,
        tune=args.tune,
        head=args.head,
        grad_checkpointing=args.grad_checkpointing,
    ).to(device)
    total = sum(p.numel() for p in model.parameters())
    print(f"{args.backbone} tune={args.tune} head={args.head}: "
          f"{model.n_trainable() / 1e6:.3f}M trainable / {total / 1e6:.1f}M total",
          flush=True)

    criterion = nn.BCEWithLogitsLoss()
    smoothing = args.label_smoothing

    def smoothed(logits, target):
        return criterion(logits, target * (1 - smoothing) + 0.5 * smoothing)

    optimizer = torch.optim.AdamW(
        model.trainable_parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    steps_per_epoch = len(loaders["train"])
    total_steps = max(steps_per_epoch * args.epochs, 1)
    warmup_steps = max(int(total_steps * args.warmup_frac), 1)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{args.tag}" if args.tag else ""
    run_name = f"v2_{args.backbone}_{args.tune}{suffix}_{stamp}"
    out_root = Path(args.out)
    ckpt_dir = out_root / "checkpoints" / run_name
    run_dir = out_root / "runs" / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "log.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(
            ["epoch", "train_loss", "train_acc", "val_macro_auc", "val_pooled_auc",
             "unseen_macro_auc", "unseen_pooled_auc", "seconds"]
        )

    best_score = -1.0
    best_epoch = -1
    best_report: dict = {}
    stale = 0
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_metrics = train_one_epoch(
            model, loaders["train"], smoothed, optimizer, scheduler, scaler,
            device, amp, args.clip, args.feature_noise, args.feature_mixup,
        )
        val = run_eval(model, loaders["val"], device, amp)
        unseen = run_eval(model, loaders["val_unseen"], device, amp)
        elapsed = time.time() - epoch_start

        print(
            f"epoch {epoch:>2}/{args.epochs} loss {train_metrics['loss']:.4f} "
            f"acc {train_metrics['acc']:.4f} | val macro {val['macro_auc']:.4f} "
            f"| UNSEEN macro {unseen['macro_auc']:.4f} pooled "
            f"{unseen['pooled_auc']:.4f} | {elapsed:.0f}s",
            flush=True,
        )
        print("   unseen per-method: " + "  ".join(
            f"{k}={v:.3f}" for k, v in sorted(unseen["per_method"].items())
        ), flush=True)

        with open(log_path, "a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow([
                epoch, round(train_metrics["loss"], 6), round(train_metrics["acc"], 6),
                round(val["macro_auc"], 6), round(val["pooled_auc"], 6),
                round(unseen["macro_auc"], 6), round(unseen["pooled_auc"], 6),
                round(elapsed, 1),
            ])

        score = unseen["macro_auc"]
        if not math.isnan(score) and score > best_score:
            best_score = score
            best_epoch = epoch
            best_report = {"val": val, "unseen": unseen}
            stale = 0
            torch.save(model.state_dict(), ckpt_dir / "best.pth")
            print(f"  new best unseen macro AUC {best_score:.4f}", flush=True)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stop at epoch {epoch}", flush=True)
                break

    config = {
        "version": 2,
        "backbone": args.backbone,
        "tune": args.tune,
        "head": args.head,
        "image_size": args.image_size,
        "mean": list(mean),
        "std": list(std),
        "augment": args.augment,
        "epochs_run": epoch,
        "best_epoch": best_epoch,
        "best_unseen_macro_auc": best_score,
        "train_methods": sorted(protocol.HELD_IN),
        "val_unseen_methods": sorted(protocol.VAL_UNSEEN),
        "args": vars(args),
        "train_minutes": round((time.time() - started) / 60, 1),
    }
    with open(ckpt_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    with open(run_dir / "best_report.json", "w", encoding="utf-8") as handle:
        json.dump(best_report, handle, indent=2)

    print(f"\nbest epoch {best_epoch} unseen macro AUC {best_score:.4f}")
    print(f"checkpoint: {ckpt_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
