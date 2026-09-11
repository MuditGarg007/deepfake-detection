import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from .. import config, database
from ..models import add_has_video
from ..schemas import AnalysisOut
from ..services import video_processor
from ..services.video_processor import NoFacesError, UnreadableVideoError

logger = logging.getLogger(__name__)

router = APIRouter(tags=["analysis"])

ALLOWED_EXTENSIONS = [".mp4", ".avi", ".mov"]
CHUNK_BYTES = 1024 * 1024

INSERT_ANALYSIS = """
INSERT INTO analyses (
    filename, storage_path, fake_probability, status,
    suspicious_start, suspicious_end, frame_scores
) VALUES (%s, %s, %s, %s, %s, %s, %s)
RETURNING *
"""


def save_upload(upload):
    suffix = Path(upload.filename or "").suffix.lower()
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    destination = config.UPLOAD_DIR / (uuid.uuid4().hex + suffix)

    limit = config.MAX_UPLOAD_MB * 1024 * 1024
    written = 0
    try:
        with open(destination, "wb") as handle:
            while True:
                chunk = upload.file.read(CHUNK_BYTES)
                if not chunk:
                    break
                written += len(chunk)
                if written > limit:
                    raise HTTPException(
                        status_code=413,
                        detail=f"file exceeds the {config.MAX_UPLOAD_MB} MB limit",
                    )
                handle.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise

    if written == 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="file is empty")
    return destination


@router.post("/analyze", response_model=AnalysisOut, status_code=201)
def analyze(file: UploadFile = File(...)):
    filename = file.filename or ""
    if Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="unsupported file type. allowed: " + ", ".join(ALLOWED_EXTENSIONS),
        )

    path = save_upload(file)
    try:
        result = video_processor.process_video(path, filename)
    except (UnreadableVideoError, NoFacesError) as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception:
        path.unlink(missing_ok=True)
        logger.exception("analysis failed for %s", filename)
        raise HTTPException(status_code=500, detail="analysis failed")

    frame_scores = [
        {"timestamp": score.timestamp, "fake_probability": score.fake_probability}
        for score in result.frame_scores
    ]
    row = database.fetch_one(
        INSERT_ANALYSIS,
        (
            result.filename,
            str(path),
            result.fake_probability,
            result.status,
            result.suspicious_start,
            result.suspicious_end,
            database.to_json(frame_scores),
        ),
    )
    return add_has_video(row)
