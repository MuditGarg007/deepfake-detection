"""Choose the operating point against a false-positive budget, not by guessing 0.5.

Every number in the v2 evaluation is reported at threshold 0.5, which is not a
decision — it is the arbitrary midpoint of a sigmoid whose scale is a learned
parameter. What a product needs is the opposite direction: *state how often
calling a genuine video fake is acceptable, then find the score that delivers
it.*

Three things follow from that framing, and all three are why this is a separate
pass rather than a line in ``evaluate_v2.py``:

1.  **The threshold is picked on val, never on test.** Held-out methods are the
    honest generalization number; tuning a threshold against them spends that
    honesty. Reals from the ``val`` split, fakes from ``val`` only.
2.  **It is picked under deployment conditions.** A threshold chosen on
    pristine PNG is wrong on codec output — measured: the ViT-B model's
    false-positive rate goes 0.086 clean to 0.493 at JPEG q10 while its AUC
    only falls 0.82 to 0.69. The score distribution shifts bodily. So the real
    pool is scored under a *mixture* of clean and degraded conditions and the
    quantile is taken over the mixture, which puts the operating point where
    the deployed distribution actually sits.
3.  **Video and frame get separate thresholds.** The backend decides a verdict
    from the mean of a video's frame scores and separately highlights
    individual frames; averaging shrinks the spread, so a frame threshold
    applied to a video mean is not the same budget.

The result is written into the checkpoint's ``config.json`` under
``operating_point``, which ``backend/services/detector.py`` reads — so pointing
``MODEL_DIR`` at a new checkpoint moves the thresholds with it instead of
leaving the v1 constants in place.

Usage::

    python machine-learning/calibrate_v2.py --checkpoint <ckpt> --amp --write
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ML_DIR = Path(__file__).resolve().parent
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from augment import build_corruption, build_eval_transform  # noqa: E402
from dataset_v2 import ForgeryDataset  # noqa: E402
from evaluate_v2 import load_any, score  # noqa: E402

# What a frame looks like by the time it reaches the detector: some arrive
# untouched, most have been through a codec at least once, some have been
# rescaled. Equal weight on each is a deliberate simplification — the point is
# that the real pool is not pristine, not that this exact mixture is the true
# deployment distribution.
DEPLOYMENT_CONDITIONS = ("clean", "jpeg_q40", "jpeg_q20", "downscale_0.5")


def video_of(paths: pd.Series) -> pd.Series:
    """Video id for a manifest path — the same grouping the backend aggregates on."""
    return paths.map(lambda p: str(Path(p).parent))


def score_conditions(model, data: str, selection: pd.DataFrame, mean, std, size,
                     conditions: tuple[str, ...], device: torch.device, amp: bool,
                     batch_size: int, num_workers: int) -> dict[str, np.ndarray]:
    probs = {}
    for condition in conditions:
        transform = (
            build_eval_transform(size, mean, std)
            if condition == "clean"
            else build_corruption(condition, size, mean, std)
        )
        loader = DataLoader(
            ForgeryDataset(data, selection, None, transform),
            batch_size=batch_size, shuffle=False, num_workers=num_workers,
            pin_memory=device.type == "cuda",
        )
        probs[condition] = score(model, loader, device, amp)
        print(f"  scored {condition}", flush=True)
    return probs


def pooled(values: dict[str, np.ndarray], conditions: tuple[str, ...]) -> np.ndarray:
    return np.concatenate([values[c] for c in conditions])


def video_means(probs: np.ndarray, videos: np.ndarray) -> np.ndarray:
    """Mean score per video — what ``video_processor.aggregate`` computes."""
    frame = pd.DataFrame({"video": videos, "prob": probs})
    return frame.groupby("video")["prob"].mean().to_numpy()


def threshold_at(real_scores: np.ndarray, budget: float) -> float:
    """Lowest score whose false-positive rate on ``real_scores`` is <= ``budget``."""
    if budget <= 0:
        return 1.0
    if budget >= 1:
        return 0.0
    return float(np.quantile(real_scores, 1.0 - budget))


def report(name: str, real: np.ndarray, fake: np.ndarray,
           thresholds: dict[str, float]) -> dict:
    out = {"n_real": int(len(real)), "n_fake": int(len(fake))}
    print(f"\n  {name}: {len(real)} real, {len(fake)} fake")
    for label, value in thresholds.items():
        fpr = float((real >= value).mean())
        recall = float((fake >= value).mean()) if len(fake) else float("nan")
        out[label] = {"threshold": value, "real_fpr": fpr, "fake_recall": recall}
        print(f"    {label:<16} t={value:.4f}  real FPR {fpr:.4f}  "
              f"fake recall {recall:.4f}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="data/processed_v2")
    parser.add_argument("--split", default="val",
                        help="never 'test': the held-out methods are the "
                             "honest number and tuning against them spends it")
    parser.add_argument("--conditions", nargs="*", default=list(DEPLOYMENT_CONDITIONS))
    parser.add_argument("--fpr-suspicious", type=float, default=0.10,
                        help="share of genuine videos allowed to reach SUSPICIOUS")
    parser.add_argument("--fpr-high", type=float, default=0.01,
                        help="share of genuine videos allowed to reach HIGH_RISK")
    parser.add_argument("--fpr-frame", type=float, default=0.05,
                        help="share of genuine frames allowed to be highlighted")
    parser.add_argument("--max-real", type=int, default=4000)
    parser.add_argument("--max-per-method", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--write", action="store_true",
                        help="write operating_point into the checkpoint config")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.split == "test":
        raise SystemExit("refusing to calibrate on the test split — see --split")

    device = torch.device(args.device)
    amp = args.amp and device.type == "cuda"
    model, config, mean, std, size = load_any(args.checkpoint, device)
    name = Path(args.checkpoint).name
    print(f"checkpoint {name}  (v{config.get('version', 1)}, "
          f"{config.get('backbone', config.get('model'))})", flush=True)

    manifest = pd.read_csv(Path(args.data) / "manifest.csv", low_memory=False)
    frame = manifest[manifest["split"] == args.split]
    rng = np.random.default_rng(0)

    def cap(group: pd.DataFrame, limit: int) -> pd.DataFrame:
        if len(group) <= limit:
            return group
        return group.iloc[rng.choice(len(group), limit, replace=False)]

    reals = cap(frame[frame["label"] == "real"], args.max_real)
    # Held-out fakes are excluded here as well as from the threshold search:
    # the recall printed beside the threshold should be a number this procedure
    # did not get to see, and mixing held-out methods into it would make the
    # calibration report quietly optimistic.
    fakes = frame[(frame["label"] == "fake") & (frame["group"] != "heldout")]
    fakes = pd.concat([cap(group, args.max_per_method)
                       for _, group in fakes.groupby("method")])
    selection = pd.concat([reals, fakes]).reset_index(drop=True)
    is_real = (selection["label"] == "real").to_numpy()
    videos = video_of(selection["path"]).to_numpy()
    print(f"split={args.split}: {len(selection)} frames "
          f"({int(is_real.sum())} real), "
          f"{selection['method'].nunique() - 1} manipulations", flush=True)

    conditions = tuple(args.conditions)
    probs = score_conditions(model, args.data, selection, mean, std, size,
                             conditions, device, amp, args.batch_size,
                             args.num_workers)

    results: dict[str, dict] = {}
    chosen: dict[str, float] = {}

    for label, subset in (("clean", ("clean",)), ("deployment", conditions)):
        real_frames = pooled({c: probs[c][is_real] for c in subset}, subset)
        fake_frames = pooled({c: probs[c][~is_real] for c in subset}, subset)
        real_videos = np.concatenate(
            [video_means(probs[c][is_real], videos[is_real]) for c in subset]
        )
        fake_videos = np.concatenate(
            [video_means(probs[c][~is_real], videos[~is_real]) for c in subset]
        )

        video_thresholds = {
            "risk_suspicious": threshold_at(real_videos, args.fpr_suspicious),
            "risk_high": threshold_at(real_videos, args.fpr_high),
        }
        frame_thresholds = {
            "frame_threshold": threshold_at(real_frames, args.fpr_frame),
        }
        print(f"\n--- {label} ({', '.join(subset)})")
        results[label] = {
            "conditions": list(subset),
            "video": report("video level", real_videos, fake_videos,
                            video_thresholds),
            "frame": report("frame level", real_frames, fake_frames,
                            frame_thresholds),
        }
        if label == "deployment":
            chosen = {**video_thresholds, **frame_thresholds}

    # Per-condition false-positive rate at the chosen thresholds. This is the
    # table that says whether the operating point survives compression, which
    # is the failure the whole exercise exists to fix.
    print("\nfalse-positive rate at the chosen thresholds, by condition:")
    per_condition = {}
    for condition in conditions:
        real_videos = video_means(probs[condition][is_real], videos[is_real])
        real_frames = probs[condition][is_real]
        entry = {
            "video_fpr_suspicious": float(
                (real_videos >= chosen["risk_suspicious"]).mean()),
            "video_fpr_high": float((real_videos >= chosen["risk_high"]).mean()),
            "frame_fpr": float((real_frames >= chosen["frame_threshold"]).mean()),
        }
        per_condition[condition] = entry
        print(f"  {condition:<15} suspicious {entry['video_fpr_suspicious']:.4f}  "
              f"high {entry['video_fpr_high']:.4f}  "
              f"frame {entry['frame_fpr']:.4f}")

    operating_point = {
        "selected_on": {
            "split": args.split,
            "conditions": list(conditions),
            "fpr_suspicious": args.fpr_suspicious,
            "fpr_high": args.fpr_high,
            "fpr_frame": args.fpr_frame,
        },
        "risk_suspicious": chosen["risk_suspicious"],
        "risk_high": chosen["risk_high"],
        "frame_threshold": chosen["frame_threshold"],
        "measured": results,
        "per_condition_real_fpr": per_condition,
    }

    print(f"\nchosen: risk_suspicious={chosen['risk_suspicious']:.4f}  "
          f"risk_high={chosen['risk_high']:.4f}  "
          f"frame_threshold={chosen['frame_threshold']:.4f}")

    if args.write:
        config_path = Path(args.checkpoint) / "config.json"
        with open(config_path, encoding="utf-8") as handle:
            stored = json.load(handle)
        stored["operating_point"] = operating_point
        with open(config_path, "w", encoding="utf-8") as handle:
            json.dump(stored, handle, indent=2)
        print(f"wrote operating_point into {config_path}")
    else:
        print("(dry run — pass --write to store it in the checkpoint config)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
