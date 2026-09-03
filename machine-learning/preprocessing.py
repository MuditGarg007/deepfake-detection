"""T3 — build the face-crop dataset from raw FaceForensics++ videos.

Pipeline (docs/ml-roadmap.md §T3, docs/plan.md §5):

1. Split at the **identity** level 70/15/15 (see ``identity_group``) so neither
   frames nor source identities are shared between splits.
2. Sample frames at ``--fps`` (default 5), capped at ``--max-frames`` per video
   (uniform subsample when the video is longer).
3. Detect faces with MTCNN; keep detections at confidence >= ``--conf`` and take
   the largest box when a frame has several faces.
4. Crop with a ``--margin`` px border and resize to 224x224.
5. Write JPEGs to ``data/processed/{split}/{label}/`` and append to
   ``data/processed/manifest.csv``.

Usage::

    python machine-learning/preprocessing.py --raw data/raw --out data/processed \
        --fps 5 --max-frames 50 --margin 20 --conf 0.95 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import queue
import random
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

CROP_SIZE = 224
# Raw class directory name -> dataset label. The Hugging Face mirror of FF++ c23
# uses these folder names; --real-dir / --fake-dir override them.
DEFAULT_REAL_DIR = "Real"
DEFAULT_FAKE_DIR = "Deepfakes"

# OpenCV spawns its own threads per VideoCapture; the decoder pool below already
# provides parallelism, so keep each capture single-threaded.
cv2.setNumThreads(1)


def find_videos(raw_dir: Path, real_dir: str, fake_dir: str) -> dict[str, list[Path]]:
    """Return ``{"real": [...], "fake": [...]}`` sorted video paths."""
    videos: dict[str, list[Path]] = {}
    for label, name in (("real", real_dir), ("fake", fake_dir)):
        candidates = [p for p in raw_dir.rglob(name) if p.is_dir()]
        if not candidates:
            raise SystemExit(f"no '{name}' directory found under {raw_dir}")
        files: list[Path] = []
        for directory in candidates:
            files.extend(directory.rglob("*.mp4"))
        if not files:
            raise SystemExit(f"no .mp4 files found under {name}/ in {raw_dir}")
        videos[label] = sorted(files)
    return videos


def identity_group(path: Path) -> str:
    """Source-identity key for a FaceForensics++ video.

    Real clips are named ``033.mp4``; the Deepfakes clip that swaps a face onto
    that clip is ``033_097.mp4``. Both map to group ``033``, so splitting on this
    key keeps every clip of one identity inside a single split — a per-video
    split alone would leak an identity from train into test.
    """
    return path.stem.split("_")[0]


def split_videos(
    paths: list[Path], seed: int, ratios: tuple[float, float, float]
) -> dict[Path, str]:
    """Assign each video to train/val/test, splitting whole identity groups."""
    groups: dict[str, list[Path]] = {}
    for path in paths:
        groups.setdefault(identity_group(path), []).append(path)

    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    n = len(keys)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])

    assignment: dict[Path, str] = {}
    for i, key in enumerate(keys):
        if i < n_train:
            split = "train"
        elif i < n_train + n_val:
            split = "val"
        else:
            split = "test"
        for path in groups[key]:
            assignment[path] = split
    return assignment


def sample_indices(total: int, src_fps: float, target_fps: float, cap: int) -> list[int]:
    """Frame indices at ``target_fps``, uniformly thinned to at most ``cap``."""
    step = max(1, round(src_fps / target_fps)) if src_fps > 0 else 1
    indices = list(range(0, total, step))
    if len(indices) > cap:
        picks = [round(i * (len(indices) - 1) / (cap - 1)) for i in range(cap)]
        indices = [indices[p] for p in dict.fromkeys(picks)]
    return indices


def decode_video(path: Path, target_fps: float, cap: int) -> list[tuple[int, np.ndarray]]:
    """Return ``[(frame_index, rgb_frame), ...]`` sampled from the video."""
    capture = cv2.VideoCapture(str(path))
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        src_fps = capture.get(cv2.CAP_PROP_FPS)
        if total <= 0:
            return []
        wanted = set(sample_indices(total, src_fps, target_fps, cap))
        frames: list[tuple[int, np.ndarray]] = []
        idx = 0
        last_wanted = max(wanted) if wanted else -1
        while idx <= last_wanted:
            # grab() skips decoding for frames we do not need.
            if not capture.grab():
                break
            if idx in wanted:
                ok, frame = capture.retrieve()
                if ok:
                    frames.append((idx, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            idx += 1
        return frames
    finally:
        capture.release()


def prefetch(paths: list[Path], target_fps: float, cap: int, workers: int, depth: int):
    """Yield ``(path, frames)`` while a thread pool decodes ahead of the GPU."""
    results: queue.Queue = queue.Queue(maxsize=depth)
    work: queue.Queue = queue.Queue()
    for path in paths:
        work.put(path)

    def worker() -> None:
        while True:
            try:
                path = work.get_nowait()
            except queue.Empty:
                return
            try:
                results.put((path, decode_video(path, target_fps, cap)))
            except Exception as exc:  # a corrupt video must not kill the run
                results.put((path, exc))

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(workers)]
    for thread in threads:
        thread.start()
    for _ in paths:
        yield results.get()
    for thread in threads:
        thread.join()


def largest_face(boxes: np.ndarray, probs: np.ndarray, conf: float) -> np.ndarray | None:
    """Largest box among detections above ``conf``, or None."""
    keep = [i for i, p in enumerate(probs) if p is not None and p >= conf]
    if not keep:
        return None
    areas = [
        (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]) for i in keep
    ]
    return boxes[keep[int(np.argmax(areas))]]


def crop_face(frame: np.ndarray, box: np.ndarray, margin: int) -> np.ndarray | None:
    """Crop ``box`` with ``margin`` px of context and resize to 224x224."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = (int(round(float(v))) for v in box)
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(width, x2 + margin)
    y2 = min(height, y2 + margin)
    if x2 - x1 < 16 or y2 - y1 < 16:
        return None
    crop = frame[y1:y2, x1:x2]
    interp = cv2.INTER_AREA if crop.shape[0] > CROP_SIZE else cv2.INTER_CUBIC
    return cv2.resize(crop, (CROP_SIZE, CROP_SIZE), interpolation=interp)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--out", default="data/processed")
    parser.add_argument("--real-dir", default=DEFAULT_REAL_DIR)
    parser.add_argument("--fake-dir", default=DEFAULT_FAKE_DIR)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--max-frames", type=int, default=50)
    parser.add_argument("--margin", type=int, default=20)
    parser.add_argument("--conf", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-videos", type=int, default=None, help="cap videos per class"
    )
    parser.add_argument("--batch-size", type=int, default=16, help="MTCNN batch size")
    parser.add_argument("--workers", type=int, default=4, help="video decoder threads")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    args = parser.parse_args()

    raw_dir = Path(args.raw)
    out_dir = Path(args.out)
    videos = find_videos(raw_dir, args.real_dir, args.fake_dir)
    if args.max_videos:
        videos = {k: v[: args.max_videos] for k, v in videos.items()}

    labels: dict[Path, str] = {}
    for label, paths in videos.items():
        labels.update({p: label for p in paths})
        print(f"{label}: {len(paths)} videos")

    # One split over both classes at once, so an identity present as both a real
    # and a fake clip cannot straddle two splits.
    splits = split_videos(list(labels), args.seed, (0.70, 0.15, 0.15))

    for split in ("train", "val", "test"):
        for label in ("real", "fake"):
            (out_dir / split / label).mkdir(parents=True, exist_ok=True)

    from facenet_pytorch import MTCNN

    mtcnn = MTCNN(keep_all=True, min_face_size=40, device=args.device)

    manifest_path = out_dir / "manifest.csv"
    ordered = sorted(labels, key=lambda p: (labels[p], p.name))
    total_frames = 0
    total_crops = 0
    per_split: dict[tuple[str, str], int] = {}
    failed: list[str] = []

    with open(manifest_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["video", "split", "label", "frame_idx", "path"])

        # Prefetch depth == worker count: decoded frames are raw uint8 arrays, so a
        # deeper queue costs hundreds of MB of RAM per queued 1080p video.
        stream = prefetch(ordered, args.fps, args.max_frames, args.workers, args.workers)
        for path, frames in tqdm(stream, total=len(ordered), desc="videos"):
            if isinstance(frames, Exception):
                failed.append(f"{path.name}: {frames}")
                continue
            if not frames:
                failed.append(f"{path.name}: no readable frames")
                continue

            split, label = splits[path], labels[path]
            stem = path.stem
            total_frames += len(frames)

            for start in range(0, len(frames), args.batch_size):
                chunk = frames[start : start + args.batch_size]
                images = [frame for _, frame in chunk]
                with torch.no_grad():
                    batch_boxes, batch_probs = mtcnn.detect(images)

                for (frame_idx, frame), boxes, probs in zip(chunk, batch_boxes, batch_probs):
                    if boxes is None or len(boxes) == 0:
                        continue
                    box = largest_face(boxes, probs, args.conf)
                    if box is None:
                        continue
                    crop = crop_face(frame, box, args.margin)
                    if crop is None:
                        continue
                    rel = f"{split}/{label}/{stem}_{frame_idx}.jpg"
                    cv2.imwrite(
                        str(out_dir / rel),
                        cv2.cvtColor(crop, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality],
                    )
                    writer.writerow([path.name, split, label, frame_idx, rel])
                    total_crops += 1
                    per_split[(split, label)] = per_split.get((split, label), 0) + 1

    skip_rate = 1 - (total_crops / total_frames) if total_frames else 0.0
    print(f"\nframes sampled : {total_frames}")
    print(f"crops written  : {total_crops}")
    print(f"skip rate      : {skip_rate:.2%} (no face above conf {args.conf})")
    for split in ("train", "val", "test"):
        counts = {label: per_split.get((split, label), 0) for label in ("real", "fake")}
        print(f"{split:<6} real={counts['real']:<7} fake={counts['fake']}")
    if failed:
        print(f"\n{len(failed)} videos failed:")
        for line in failed[:20]:
            print("  ", line)
    print(f"\nmanifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
