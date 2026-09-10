"""Row types for the analyses table — plain dataclasses, no ORM."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from .database import from_json

# Every column, in table order. History reads a narrower list (no frame_scores)
# because that column is the large one and the list view does not show it.
ALL_COLUMNS = (
    "id, filename, storage_path, fake_probability, status, "
    "suspicious_start, suspicious_end, frame_scores, created_at"
)
HISTORY_COLUMNS = "id, filename, storage_path, fake_probability, status, created_at"


@dataclass(slots=True)
class Analysis:
    """One analyses row. Fields the query did not select keep their default."""

    id: int
    filename: str
    fake_probability: float
    status: str
    created_at: datetime
    storage_path: str | None = None
    suspicious_start: float | None = None
    suspicious_end: float | None = None
    frame_scores: list = field(default_factory=list)

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "Analysis":
        created_at = row["created_at"]
        # SQLite has no date type — it hands back the stored text.
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        frame_scores = row.get("frame_scores")
        return cls(
            id=row["id"],
            filename=row["filename"],
            fake_probability=row["fake_probability"],
            status=row["status"],
            created_at=created_at,
            storage_path=row.get("storage_path"),
            suspicious_start=row.get("suspicious_start"),
            suspicious_end=row.get("suspicious_end"),
            frame_scores=[] if frame_scores is None else from_json(frame_scores),
        )

    @property
    def has_video(self) -> bool:
        """Whether the source video is still on disk and can be streamed back."""
        return self.storage_path is not None and Path(self.storage_path).is_file()
