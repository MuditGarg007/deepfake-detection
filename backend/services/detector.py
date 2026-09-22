import logging
import os
import sys
from pathlib import Path

from .. import config

logger = logging.getLogger(__name__)

ML_DIR = config.PROJECT_DIR / "machine-learning"

MODEL = None
INFERENCE = None


def find_checkpoint(directory):
    if (directory / "config.json").is_file():
        return directory
    runs = [child for child in directory.glob("*/") if (child / "config.json").is_file()]
    if not runs:
        return directory
    latest = max(runs, key=lambda child: child.stat().st_mtime)
    logger.info("MODEL_DIR is a parent directory - using latest run %s", latest.name)
    return latest


def load_model(model_dir):
    global MODEL, INFERENCE

    MODEL = None
    INFERENCE = None

    if str(ML_DIR) not in sys.path:
        sys.path.append(str(ML_DIR))

    try:
        import inference
    except Exception as exc:
        logger.warning("inference module unavailable (%s) - using stub predictor", exc)
        return

    directory = Path(model_dir)
    if not directory.is_absolute():
        directory = config.PROJECT_DIR / directory
    directory = find_checkpoint(directory)

    if not (directory / "config.json").is_file() or not (directory / "best.pth").is_file():
        logger.warning("Checkpoint not found at %s - using stub predictor", directory)
        return

    MODEL = inference.load_model(str(directory))
    INFERENCE = inference
    logger.info("Model loaded from %s", directory)


def crop_style():
    """Which face-crop convention the loaded checkpoint was trained on.

    v1 checkpoints saw MTCNN boxes padded by a fixed margin; v2 checkpoints saw
    DeepfakeBench-aligned square crops. ``video_processor.crop_face`` asks here
    so swapping MODEL_DIR between generations needs no other change. With no
    model loaded the stub predictor ignores the crop anyway.
    """
    config = getattr(MODEL, "config", None) or {}
    return "aligned" if config.get("version") == 2 else "margin"


def risk_bands():
    """Score thresholds for the loaded checkpoint, falling back to the config.

    A threshold is a property of a model, not of a deployment: it is the point
    on *that* model's score distribution where the false-positive rate hits an
    agreed budget. ``machine-learning/calibrate_v2.py`` measures it on the val
    split under deployment-like degradation and writes it into the
    checkpoint's ``config.json``, so pointing MODEL_DIR at a new checkpoint
    moves the operating point with the weights.

    The checkpoint therefore wins over ``RISK_SUSPICIOUS`` / ``RISK_HIGH`` /
    ``FRAME_THRESHOLD``. Those are v1-era constants that ``backend/.env``
    already sets to 0.4 / 0.7 / 0.7, so letting the environment win would mean
    a calibrated checkpoint silently never takes effect — which is the bug this
    function exists to remove, not a form of it. They remain the fallback for a
    checkpoint that carries no operating point, and ``RISK_BANDS_SOURCE=env``
    forces them back for an operator who means it.
    """
    bands = {
        "risk_suspicious": config.RISK_SUSPICIOUS,
        "risk_high": config.RISK_HIGH,
        "frame_threshold": config.FRAME_THRESHOLD,
    }
    if os.getenv("RISK_BANDS_SOURCE", "").lower() == "env":
        return bands
    stored = (getattr(MODEL, "config", None) or {}).get("operating_point") or {}
    for key in bands:
        if key in stored:
            bands[key] = float(stored[key])
    return bands


def predict(crop):
    if MODEL is None or INFERENCE is None:
        return 0.5
    return float(INFERENCE.predict(MODEL, crop))


def predict_batch(crops):
    if not crops:
        return []
    if MODEL is None or INFERENCE is None:
        return [0.5] * len(crops)
    return [float(p) for p in INFERENCE.predict(MODEL, list(crops))]
