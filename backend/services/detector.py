"""Detector service (T5) — wraps ``machine-learning/inference.py``.

Loads the Phase 1 checkpoint at startup and exposes ``predict(crop) -> float``
(0-1 fake probability). If the Phase 1 module or checkpoint is missing, logs a
warning and runs in stub mode where ``predict`` returns a fixed ``0.5`` so the
routes and frontend can still be developed.
"""

import logging
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[1]
_PROJECT_DIR = _BACKEND_DIR.parent
_ML_DIR = _PROJECT_DIR / "machine-learning"

logger = logging.getLogger(__name__)

MODEL = None
# The Phase 1 module is imported lazily inside load_model (it may not exist
# yet), so keep the module object here for predict() to use afterwards.
INFERENCE = None


def _resolve_checkpoint(directory: Path) -> Path:
    """Allow MODEL_DIR to point at either a run directory or their parent.

    Training writes each run to ``checkpoints/<model>_<timestamp>/``, so the
    default ``MODEL_DIR`` (``machine-learning/checkpoints``) is the parent. When
    it is, pick the most recently written run inside it.
    """
    if (directory / "config.json").is_file():
        return directory
    runs = [child for child in directory.glob("*/") if (child / "config.json").is_file()]
    if not runs:
        return directory
    latest = max(runs, key=lambda child: child.stat().st_mtime)
    logger.info("MODEL_DIR is a parent directory — using latest run %s", latest.name)
    return latest


def load_model(model_dir: str) -> None:
    """Load the Phase 1 checkpoint, or fall back to stub mode."""
    global MODEL, INFERENCE

    MODEL = None
    INFERENCE = None

    # Phase 1 may not be built yet — make the service importable if present.
    if str(_ML_DIR) not in sys.path:
        sys.path.append(str(_ML_DIR))

    try:
        import inference
    except Exception as exc:  # module absent or broken -> stub
        logger.warning("inference module unavailable (%s) — using stub predictor", exc)
        return

    directory = Path(model_dir)
    if not directory.is_absolute():
        directory = _PROJECT_DIR / directory
    directory = _resolve_checkpoint(directory)

    if not (directory / "config.json").is_file() or not (directory / "best.pth").is_file():
        logger.warning(
            "Checkpoint not found at %s — using stub predictor (predict returns 0.5)",
            directory,
        )
        return

    MODEL = inference.load_model(str(directory))
    INFERENCE = inference
    logger.info("Model loaded from %s", directory)


def predict(crop) -> float:
    """Return a 0-1 fake probability for a face crop. Stub returns 0.5."""
    if MODEL is None or INFERENCE is None:
        return 0.5
    return float(INFERENCE.predict(MODEL, crop))


def predict_batch(crops: list) -> list[float]:
    """Score a whole video's crops in one GPU batch. Stub returns 0.5 each."""
    if not crops:
        return []
    if MODEL is None or INFERENCE is None:
        return [0.5] * len(crops)
    return [float(p) for p in INFERENCE.predict(MODEL, list(crops))]
