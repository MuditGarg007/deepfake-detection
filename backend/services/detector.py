import logging
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
