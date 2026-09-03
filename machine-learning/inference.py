"""T8 — inference API consumed by Phase 2 (``backend/services/detector.py``).

::

    from inference import load_model, predict

    model = load_model("machine-learning/checkpoints/efficientnet_b0_<ts>")
    prob  = predict(model, "data/processed/test/fake/xxx.jpg")   # 0.0-1.0 fake

``predict`` accepts a file path, a PIL image, or an RGB ``numpy`` array (the
form ``video_processor`` passes after cropping a face), and also a list of any
of those for batched scoring.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ML_DIR = Path(__file__).resolve().parent
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from dataset import build_transform  # noqa: E402
from models import get_model  # noqa: E402

_TRANSFORM = build_transform(train=False)


def load_model(checkpoint_dir: str | Path, device: str | torch.device | None = None):
    """Rebuild the architecture from ``config.json`` and load ``best.pth``.

    Returns an eval-mode module with ``.device`` attached so ``predict`` knows
    where to put its inputs.
    """
    directory = Path(checkpoint_dir)
    config_path = directory / "config.json"
    weights_path = directory / "best.pth"
    if not config_path.is_file():
        raise FileNotFoundError(f"{config_path} not found")
    if not weights_path.is_file():
        raise FileNotFoundError(f"{weights_path} not found")

    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # pretrained=False: the fine-tuned weights below replace the ImageNet ones,
    # so there is no reason to download them.
    model = get_model(config["model"], pretrained=False)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device).eval()
    model.device = device
    model.config = config
    return model


def _to_pil(image) -> Image.Image:
    """Normalize a path / PIL image / RGB ndarray into an RGB PIL image."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    if isinstance(image, np.ndarray):
        array = image
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        if array.ndim == 2:
            array = np.stack([array] * 3, axis=-1)
        return Image.fromarray(array[:, :, :3], mode="RGB")
    if isinstance(image, torch.Tensor):
        return _to_pil(image.detach().cpu().numpy())
    raise TypeError(f"unsupported image type: {type(image)}")


@torch.no_grad()
def predict(model, image) -> float | list[float]:
    """Fake probability in [0, 1]. A list input returns a list of floats."""
    batched = isinstance(image, (list, tuple))
    images = list(image) if batched else [image]
    if not images:
        return []

    device = getattr(model, "device", next(model.parameters()).device)
    tensor = torch.stack([_TRANSFORM(_to_pil(img)) for img in images]).to(device)
    probabilities = torch.sigmoid(model(tensor).squeeze(1).float()).cpu().numpy()
    return [float(p) for p in probabilities] if batched else float(probabilities[0])


def _smoke_test(checkpoint_dir: str, data_dir: str, limit: int) -> int:
    """Score ``limit`` test crops per class and report the mean probability."""
    import pandas as pd

    model = load_model(checkpoint_dir)
    frame = pd.read_csv(Path(data_dir) / "manifest.csv")
    frame = frame[frame["split"] == "test"]
    if frame.empty:
        raise SystemExit("manifest has no test rows")

    failures = 0
    for label in ("real", "fake"):
        rows = frame[frame["label"] == label].head(limit)
        paths = [Path(data_dir) / p for p in rows["path"]]
        probs = predict(model, paths)
        if not all(0.0 <= p <= 1.0 for p in probs):
            print(f"FAIL: {label} probabilities outside [0, 1]")
            failures += 1
        mean = sum(probs) / len(probs)
        print(f"{label:<5} n={len(probs):<4} mean fake prob = {mean:.4f}")
        if label == "fake" and mean < 0.5:
            print("FAIL: mean fake probability below 0.5 on fake crops")
            failures += 1
        if label == "real" and mean > 0.5:
            print("FAIL: mean fake probability above 0.5 on real crops")
            failures += 1

    # Single-image and ndarray paths must agree with the batched path.
    single_path = Path(data_dir) / frame.iloc[0]["path"]
    single = predict(model, single_path)
    array = predict(model, np.array(Image.open(single_path).convert("RGB")))
    if abs(single - array) > 1e-4:
        print(f"FAIL: path {single:.6f} and ndarray {array:.6f} disagree")
        failures += 1
    else:
        print(f"path/ndarray agree: {single:.4f}")

    print("SMOKE TEST PASSED" if failures == 0 else f"SMOKE TEST FAILED ({failures})")
    return 1 if failures else 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="inference smoke test")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--limit", type=int, default=100)
    parsed = parser.parse_args()
    sys.exit(_smoke_test(parsed.checkpoint, parsed.data, parsed.limit))
