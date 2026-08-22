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


def load_model(model_dir: str) -> None:
    """Load the Phase 1 checkpoint, or fall back to stub mode."""
    global MODEL

    # Phase 1 may not be built yet — make the service importable if present.
    if str(_ML_DIR) not in sys.path:
        sys.path.append(str(_ML_DIR))

    try:
        import inference
    except Exception as exc:  # module absent or broken -> stub
        logger.warning("inference module unavailable (%s) — using stub predictor", exc)
        MODEL = None
        return

    directory = Path(model_dir)
    if not directory.is_absolute():
        directory = _PROJECT_DIR / directory

    if not (directory / "config.json").is_file() or not (directory / "best.pth").is_file():
        logger.warning(
            "Checkpoint not found at %s — using stub predictor (predict returns 0.5)",
            directory,
        )
        MODEL = None
        return

    MODEL = inference.load_model(str(directory))
    logger.info("Model loaded from %s", directory)


def predict(crop) -> float:
    """Return a 0-1 fake probability for a face crop. Stub returns 0.5."""
    if MODEL is None:
        return 0.5
    return float(inference.predict(MODEL, crop))