"""Download the v2 (robustness) corpus: DeepfakeBench real+classic-fake frames
plus a selection of DF40 modern-manipulation frames.

Everything lands as zips in ``data/dl``; ``build_dataset_v2.py`` unpacks them.

Usage::

    python machine-learning/download_v2.py --group bench
    python machine-learning/download_v2.py --group df40_train
    python machine-learning/download_v2.py --group df40_test
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_XET_NUM_CONCURRENT_RANGE_GETS", "8")

from huggingface_hub import hf_hub_download  # noqa: E402

OUT = Path("data/dl")

# DeepfakeBench-preprocessed frames. These supply the *real* face crops, which
# must come from the same alignment pipeline as the DF40 fakes below or the
# model learns the crop convention instead of the forgery.
BENCH_REPO = "Ella4839/DeepfakeBench"
BENCH_FILES = ["Celeb-DF-v2.zip", "FaceForensics++.zip", "DFDCP.zip", "UADFV.zip"]

# DF40 manipulations kept for training (held-in methods).
DF40_TRAIN_REPO = "ManhQuangAI/DF40_train"
DF40_TRAIN_FILES = [
    # face swap
    "faceswap.zip", "blendface.zip", "facedancer.zip", "fsgan.zip", "mobileswap.zip",
    # face reenactment
    "fomm.zip", "facevid2vid.zip", "lia.zip", "tpsm.zip", "wav2lip.zip",
    # entire face synthesis
    "StyleGAN2.zip", "ddim.zip", "VQGAN.zip", "DiT.zip",
]

# DF40 manipulations held out entirely — the unseen-method test set.
DF40_TEST_REPO = "ManhQuangAI/df-40-test-full"
DF40_TEST_FILES = [
    # modern face swaps never seen in training
    "inswap.zip", "simswap.zip", "uniface.zip", "e4s.zip", "deepfacelab.zip",
    # modern talking-head / reenactment
    "sadtalker.zip", "heygen_new.zip", "hyperreenact.zip", "mcnet.zip",
    # modern generative synthesis
    "SiT.zip", "StyleGAN3.zip", "MidJourney.zip", "CollabDiff.zip",
    "starganv2.zip", "styleclip.zip", "whichfaceisreal.zip", "stargan.zip",
]

GROUPS = {
    "bench": (BENCH_REPO, BENCH_FILES),
    "df40_train": (DF40_TRAIN_REPO, DF40_TRAIN_FILES),
    "df40_test": (DF40_TEST_REPO, DF40_TEST_FILES),
}


def fetch(repo: str, filename: str, out: Path) -> Path:
    target = out / repo.replace("/", "__")
    target.mkdir(parents=True, exist_ok=True)
    done = target / filename
    if done.is_file():
        print(f"  have {filename} ({done.stat().st_size / 1e9:.2f} GB)", flush=True)
        return done
    started = time.time()
    path = Path(hf_hub_download(repo, filename, repo_type="dataset", local_dir=target))
    size = path.stat().st_size / 1e9
    elapsed = time.time() - started
    print(f"  got {filename} {size:.2f} GB in {elapsed:.0f}s "
          f"({size * 1000 / max(elapsed, 1):.0f} MB/s)", flush=True)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=sorted(GROUPS) + ["all"], default="all")
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()

    out = Path(args.out)
    groups = sorted(GROUPS) if args.group == "all" else [args.group]
    for group in groups:
        repo, files = GROUPS[group]
        print(f"=== {group} ({repo}): {len(files)} files", flush=True)
        for filename in files:
            for attempt in range(3):
                try:
                    fetch(repo, filename, out)
                    break
                except Exception as exc:  # network flake — retry
                    print(f"  !! {filename} attempt {attempt + 1}: {exc}", flush=True)
                    time.sleep(10)
            else:
                print(f"  FAILED {filename}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
