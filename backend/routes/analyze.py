"""POST /analyze (T7) — upload a video, run the pipeline, store the result."""

from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from ..config import settings
from ..database import get_db
from ..models import Analysis
from ..schemas import AnalysisOut
from ..services import video_processor
from ..services.video_processor import NoFacesError, UnreadableVideoError

logger = logging.getLogger(__name__)

router = APIRouter(tags=["analysis"])

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov"}
# Read the upload in chunks so an oversized file is rejected without ever
# holding all of it in memory.
CHUNK_BYTES = 1024 * 1024


def _save_upload(upload: UploadFile) -> Path:
    """Write the upload to UPLOAD_DIR under a uuid-prefixed name.

    Raises 413 as soon as the running total passes MAX_UPLOAD_MB.
    """
    suffix = Path(upload.filename or "").suffix.lower()
    directory = settings.upload_dir
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{uuid.uuid4().hex}{suffix}"

    limit = settings.MAX_UPLOAD_MB * 1024 * 1024
    written = 0
    try:
        with open(destination, "wb") as handle:
            while chunk := upload.file.read(CHUNK_BYTES):
                written += len(chunk)
                if written > limit:
                    raise HTTPException(
                        status_code=413,  # Content Too Large
                        detail=f"file exceeds the {settings.MAX_UPLOAD_MB} MB limit",
                    )
                handle.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise

    if written == 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="file is empty")
    return destination


@router.post("/analyze", response_model=AnalysisOut, status_code=status.HTTP_201_CREATED)
def analyze(file: UploadFile = File(...), db: Session = Depends(get_db)) -> Analysis:
    """Analyze an uploaded video and persist the result."""
    filename = file.filename or ""
    if Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"unsupported file type — allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    path = _save_upload(file)
    try:
        result = video_processor.process_video(path, filename=filename)
    except (UnreadableVideoError, NoFacesError) as exc:
        # Nothing was stored, so the file has no row to belong to — drop it.
        path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except Exception as exc:  # unexpected pipeline failure
        path.unlink(missing_ok=True)
        logger.exception("analysis failed for %s", filename)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="analysis failed"
        ) from exc

    analysis = Analysis(
        filename=result.filename,
        # Kept so GET /analysis/{id}/video can play the clip back.
        storage_path=str(path),
        fake_probability=result.fake_probability,
        status=result.status,
        suspicious_start=result.suspicious_start,
        suspicious_end=result.suspicious_end,
        frame_scores=[
            {"timestamp": s.timestamp, "fake_probability": s.fake_probability}
            for s in result.frame_scores
        ],
    )
    db.add(analysis)
    db.commit()
    db.refresh(analysis)
    return analysis
