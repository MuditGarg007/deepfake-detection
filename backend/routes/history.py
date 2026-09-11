from pathlib import Path

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import FileResponse

from .. import config, database
from ..models import add_has_video
from ..schemas import AnalysisOut, FeedbackIn, HistoryOut
from ..services import video_processor
from ..services.video_processor import NoFacesError, UnreadableVideoError

MEDIA_TYPES = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
}

router = APIRouter(tags=["analysis"])

SELECT_BY_ID = "SELECT * FROM analyses WHERE id = %s"
SELECT_HISTORY = """
SELECT id, filename, storage_path, fake_probability, status, created_at
FROM analyses ORDER BY created_at DESC, id DESC LIMIT %s
"""
UPDATE_FEEDBACK = "UPDATE analyses SET user_feedback = %s WHERE id = %s RETURNING *"
UPDATE_RESULT = """
UPDATE analyses SET
    fake_probability = %s, status = %s,
    suspicious_start = %s, suspicious_end = %s, frame_scores = %s
WHERE id = %s
RETURNING *
"""


def load(analysis_id):
    row = database.fetch_one(SELECT_BY_ID, (analysis_id,))
    if row is None:
        raise HTTPException(status_code=404, detail=f"analysis {analysis_id} not found")
    return row


def source_path(row):
    if row["storage_path"] is None:
        raise HTTPException(
            status_code=404,
            detail="this analysis was recorded without keeping the source video",
        )

    path = Path(row["storage_path"]).resolve()
    if not path.is_relative_to(config.UPLOAD_DIR.resolve()) or not path.is_file():
        raise HTTPException(status_code=404, detail="the source video is no longer available")
    return path


@router.get("/analysis/{analysis_id}", response_model=AnalysisOut)
def get_analysis(analysis_id: int):
    return add_has_video(load(analysis_id))


@router.get("/analysis/{analysis_id}/video", tags=["analysis"])
def get_analysis_video(analysis_id: int):
    row = load(analysis_id)
    path = source_path(row)
    return FileResponse(
        path,
        media_type=MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream"),
        filename=row["filename"],
        content_disposition_type="inline",
    )


@router.get(
    "/analysis/{analysis_id}/frame",
    tags=["analysis"],
    response_class=Response,
    responses={200: {"content": {"image/jpeg": {}}}},
)
def get_analysis_frame(analysis_id: int):
    row = load(analysis_id)
    if not row["frame_scores"]:
        raise HTTPException(status_code=404, detail="this analysis has no scored frames")

    path = source_path(row)
    peak = max(row["frame_scores"], key=lambda score: score["fake_probability"])
    image = video_processor.grab_frame_jpeg(path, peak["timestamp"])
    if image is None:
        raise HTTPException(
            status_code=404,
            detail="could not read that frame back from the source video",
        )

    return Response(
        content=image,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@router.post("/analysis/{analysis_id}/feedback", response_model=AnalysisOut)
def set_feedback(analysis_id: int, body: FeedbackIn):
    load(analysis_id)
    row = database.fetch_one(UPDATE_FEEDBACK, (body.label, analysis_id))
    return add_has_video(row)


@router.post("/analysis/{analysis_id}/rerun", response_model=AnalysisOut)
def rerun_analysis(analysis_id: int):
    row = load(analysis_id)
    path = source_path(row)

    try:
        result = video_processor.process_video(path, row["filename"])
    except (UnreadableVideoError, NoFacesError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    frame_scores = [
        {"timestamp": score.timestamp, "fake_probability": score.fake_probability}
        for score in result.frame_scores
    ]
    updated = database.fetch_one(
        UPDATE_RESULT,
        (
            result.fake_probability,
            result.status,
            result.suspicious_start,
            result.suspicious_end,
            database.to_json(frame_scores),
            analysis_id,
        ),
    )
    return add_has_video(updated)


@router.get("/history", response_model=list[HistoryOut])
def history(limit: int = 100):
    rows = database.fetch_all(SELECT_HISTORY, (limit,))
    return [add_has_video(row) for row in rows]
