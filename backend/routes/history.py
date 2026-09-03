"""GET /analysis/{id}, GET /analysis/{id}/video, and GET /history (T7)."""

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import settings
from ..database import get_db
from ..models import Analysis
from ..schemas import AnalysisOut, HistoryOut

# Browsers need a type they recognise; .avi is served but most will not play it.
MEDIA_TYPES = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
}

router = APIRouter(tags=["analysis"])


@router.get("/analysis/{analysis_id}", response_model=AnalysisOut)
def get_analysis(analysis_id: int, db: Session = Depends(get_db)) -> Analysis:
    analysis = db.get(Analysis, analysis_id)
    if analysis is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"analysis {analysis_id} not found"
        )
    return analysis


@router.get("/analysis/{analysis_id}/video", tags=["analysis"])
def get_analysis_video(analysis_id: int, db: Session = Depends(get_db)) -> FileResponse:
    """Stream back the video this analysis was run on, for playback in the UI.

    ``FileResponse`` answers Range requests, so the player can seek without
    pulling the whole clip first.
    """
    analysis = db.get(Analysis, analysis_id)
    if analysis is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"analysis {analysis_id} not found"
        )
    if analysis.storage_path is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="this analysis was recorded without keeping the source video",
        )

    path = Path(analysis.storage_path).resolve()
    # The path comes from our own row, but a stored value is still input: never
    # serve anything from outside the upload directory.
    if not path.is_relative_to(settings.upload_dir.resolve()) or not path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="the source video is no longer available",
        )

    return FileResponse(
        path,
        media_type=MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream"),
        filename=analysis.filename,
        content_disposition_type="inline",
    )


@router.get("/history", response_model=list[HistoryOut])
def history(
    limit: int = Query(100, ge=1, le=500), db: Session = Depends(get_db)
) -> list[Analysis]:
    """Most recent analyses first (plan §5 history table)."""
    statement = select(Analysis).order_by(Analysis.created_at.desc(), Analysis.id.desc()).limit(limit)
    return list(db.scalars(statement))
