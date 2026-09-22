"""How much does a video actually move? A frame-level detector cannot ask this.

Written to explain the largest defect in the v2 evaluation: SadTalker scores
0.416 AUC on the best model — below chance — while the same model handles
hyperreenact (0.959) and mcnet (0.827), which are the same generation family,
the same source corpus and the same split.

SadTalker animates **one still image** from audio. If the render leaves most of
the face untouched, then most pixels in any single crop are genuine, and there
is nothing for a per-frame classifier to find no matter how good its backbone
is — which is exactly what the numbers show: going from EfficientNet to ViT-B
to ViT-L moves hyperreenact 0.63 -> 0.93 -> 0.96 and moves SadTalker 0.28 ->
0.46 -> 0.42.

The measurement here is deliberately not a model. It is the per-pixel standard
deviation across several aligned crops of the same video — one scalar, no
weights, no training — plus two controls, because a feature this cheap scoring
this well is the kind of result that is usually an artifact:

*   **Sampling interval.** The dataset keeps ~8 frames per video, and if
    SadTalker's happen to be closer together in the source video its frames
    would differ less for a trivial reason. Reported per method, and the reals
    can be restricted to a matching window.
*   **An interval-free statistic.** The mean absolute difference between the
    two closest frames held has no dependence on how widely the video was
    sampled.

The result is a *specific* detector, not a general one, and the script prints
enough to see that: against genuine video the same feature scores 0.43 on
mcnet, 0.39 on simswap and 0.33 on lia — most manipulations are *more* variable
than real footage, not less. It detects still-driven rendering and nothing
else, which is why it belongs as a separate video-level signal rather than
blended into the model's score.

Usage::

    python machine-learning/temporal_v2.py --data data/processed_v2 --split test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ML_DIR = Path(__file__).resolve().parent
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

import protocol  # noqa: E402

SIZE = 128
# The lower-middle block of the crop, where a lip-sync render puts its motion.
# 40x48 of 128x128 is 11.7% of the pixels, which is the baseline any "share of
# variation in the mouth" number has to beat to mean anything.
MOUTH = (slice(80, 120), slice(40, 88))
MOUTH_AREA_SHARE = (40 * 48) / (SIZE * SIZE)


def frame_index(path: str) -> int:
    digits = "".join(c for c in Path(path).stem if c.isdigit())
    return int(digits) if digits else 0


def video_statistics(root: Path, frame: pd.DataFrame, videos: np.ndarray,
                     max_frames: int) -> pd.DataFrame:
    """Per-video temporal statistics for the given video ids."""
    rows = []
    for video in videos:
        paths = sorted(frame[frame["video"] == video]["path"])[:max_frames]
        if len(paths) < 3:
            continue
        images = [cv2.imread(str(root / p), cv2.IMREAD_GRAYSCALE) for p in paths]
        images = [cv2.resize(i, (SIZE, SIZE)) for i in images if i is not None]
        if len(images) < 3:
            continue
        stack = np.stack(images).astype(np.float32)
        per_pixel = stack.std(axis=0)
        indices = sorted(frame_index(p) for p in paths)
        rows.append({
            "video": video,
            "spread": float(per_pixel.mean()),
            "mouth_share": float(per_pixel[MOUTH].sum() / (per_pixel.sum() + 1e-9)),
            # Independent of the sampling interval: the closest pair we hold.
            "adjacent_diff": float(np.abs(stack[1] - stack[0]).mean()),
            "frame_gap": float(np.median(np.diff(indices))) if len(indices) > 1 else 0.0,
        })
    return pd.DataFrame(rows)


def auc_low_is_fake(real: np.ndarray, fake: np.ndarray) -> float:
    truth = np.concatenate([np.zeros(len(real)), np.ones(len(fake))])
    # Negated: the hypothesis is that *less* variation means fake.
    return float(roc_auc_score(truth, -np.concatenate([real, fake])))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/processed_v2")
    parser.add_argument("--split", default="test")
    parser.add_argument("--real-dataset", default="cdf",
                        help="corpus the real videos come from; cdf is the "
                             "source of the DF40 video manipulations, so it is "
                             "the matched comparison")
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--include-synthesis", action="store_true",
                        help="also score the entire-face synthesis sets, whose "
                             "'videos' are not videos; off by default")
    parser.add_argument("--max-videos", type=int, default=120)
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    root = Path(args.data)
    manifest = pd.read_csv(root / "manifest.csv", low_memory=False)
    frame = manifest[manifest["split"] == args.split].copy()
    frame["video"] = frame["path"].map(lambda p: str(Path(p).parent))
    rng = np.random.default_rng(args.seed)

    def sample(sub: pd.DataFrame, limit: int) -> np.ndarray:
        videos = sub["video"].unique()
        if len(videos) > limit:
            videos = rng.choice(videos, limit, replace=False)
        return videos

    reals = frame[(frame["label"] == "real")
                  & (frame["dataset"] == args.real_dataset)]
    real_stats = video_statistics(root, reals, sample(reals, args.max_videos),
                                  args.max_frames)
    if real_stats.empty:
        raise SystemExit(f"no real videos in split={args.split} "
                         f"dataset={args.real_dataset}")
    print(f"real ({args.real_dataset}): {len(real_stats)} videos  "
          f"spread {real_stats['spread'].median():.2f}  "
          f"frame gap {real_stats['frame_gap'].median():.0f}  "
          f"adjacent diff {real_stats['adjacent_diff'].median():.2f}\n")

    # Entire-face synthesis has no video structure — its "videos" are buckets
    # of unrelated generated images, so a within-video spread over them is a
    # spread over different people and means nothing. Including them produced a
    # column of 0.00 AUCs and median frame gaps in the thousands, which is the
    # measurement telling you it does not apply.
    methods = args.methods or sorted(
        m for m in frame[frame["label"] == "fake"]["method"].unique()
        if args.include_synthesis or protocol.family_for(m) != "synthesis"
    )
    rows = []
    print(f"{'method':<18}{'family':<14}{'n':>5}{'spread':>9}{'mouth':>8}"
          f"{'gap':>6}{'AUC':>8}{'AUC(adj)':>10}")
    for method in methods:
        sub = frame[frame["method"] == method]
        stats = video_statistics(root, sub, sample(sub, args.max_videos),
                                 args.max_frames)
        if len(stats) < 10:
            continue
        row = {
            "method": method,
            "family": protocol.family_for(method),
            "n_videos": len(stats),
            "median_spread": float(stats["spread"].median()),
            "median_mouth_share": float(stats["mouth_share"].median()),
            "median_frame_gap": float(stats["frame_gap"].median()),
            "temporal_auc": auc_low_is_fake(real_stats["spread"].to_numpy(),
                                            stats["spread"].to_numpy()),
            "adjacent_auc": auc_low_is_fake(
                real_stats["adjacent_diff"].to_numpy(),
                stats["adjacent_diff"].to_numpy()),
        }
        rows.append(row)
        print(f"{method:<18}{row['family']:<14}{row['n_videos']:>5}"
              f"{row['median_spread']:>9.2f}{row['median_mouth_share']:>8.3f}"
              f"{row['median_frame_gap']:>6.0f}{row['temporal_auc']:>8.4f}"
              f"{row['adjacent_auc']:>10.4f}")

    table = pd.DataFrame(rows).sort_values("temporal_auc", ascending=False)
    print(f"\nmouth region is {MOUTH_AREA_SHARE:.3f} of the crop, so a "
          f"'mouth' column above that means the motion is concentrated there")

    # Gap-matched control for whichever method separates best.
    best = table.iloc[0]
    lo, hi = np.percentile(
        video_statistics(
            root, frame[frame["method"] == best["method"]],
            sample(frame[frame["method"] == best["method"]], args.max_videos),
            args.max_frames,
        )["frame_gap"], [10, 90],
    )
    matched = real_stats[(real_stats["frame_gap"] >= lo)
                         & (real_stats["frame_gap"] <= hi)]
    print(f"\ngap-matched control for {best['method']}: reals with frame gap in "
          f"[{lo:.0f}, {hi:.0f}] — {len(matched)} of {len(real_stats)} videos, "
          f"median spread {matched['spread'].median():.2f} against "
          f"{best['median_spread']:.2f}")
    if len(matched) >= 5:
        stats = video_statistics(
            root, frame[frame["method"] == best["method"]],
            sample(frame[frame["method"] == best["method"]], args.max_videos),
            args.max_frames,
        )
        print(f"  AUC on gap-matched reals: "
              f"{auc_low_is_fake(matched['spread'].to_numpy(), stats['spread'].to_numpy()):.4f}")

    out = Path(args.out) if args.out else ML_DIR / "runs" / "temporal_v2.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
