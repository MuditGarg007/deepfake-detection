from pathlib import Path


def add_has_video(row):
    path = row.get("storage_path")
    row["has_video"] = path is not None and Path(path).is_file()
    return row
