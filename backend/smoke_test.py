"""T9 smoke test — drives every endpoint against a running server.

    uvicorn backend.main:app --port 8000        # in one shell
    python backend/smoke_test.py                # in another

Checks the happy path on a real and a fake sample plus the four error cases
from T8 (bad extension, oversized, unreadable video, unknown id).
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import httpx

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_REAL = PROJECT_DIR / "data/raw/Real/006.mp4"
DEFAULT_FAKE = PROJECT_DIR / "data/raw/Deepfakes/006_002.mp4"

failures: list[str] = []


def check(condition: bool, message: str) -> None:
    print(f"{'PASS' if condition else 'FAIL'}  {message}")
    if not condition:
        failures.append(message)


def upload(client: httpx.Client, path: Path, name: str | None = None) -> httpx.Response:
    with open(path, "rb") as handle:
        return client.post(
            "/analyze", files={"file": (name or path.name, handle, "video/mp4")}
        )


def _write_faceless_video(path: Path) -> Path:
    """A 2 s clip of moving noise — readable, but MTCNN finds no face in it."""
    import cv2
    import numpy as np

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (320, 240))
    rng = np.random.default_rng(0)
    for _ in range(50):
        writer.write(rng.integers(0, 256, (240, 320, 3), dtype=np.uint8))
    writer.release()
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--real", type=Path, default=DEFAULT_REAL)
    parser.add_argument("--fake", type=Path, default=DEFAULT_FAKE)
    args = parser.parse_args()

    client = httpx.Client(base_url=args.url, timeout=600.0)

    response = client.get("/health")
    check(response.status_code == 200, f"GET /health -> {response.status_code}")
    health = response.json()
    check(health.get("status") == "ok", f"health status = {health.get('status')}")
    if not health.get("model_loaded"):
        print("WARN  server is in stub mode (no checkpoint) — scores will all be 0.5")

    # --- happy path: real sample -------------------------------------------
    response = upload(client, args.real)
    check(response.status_code == 201, f"POST /analyze (real) -> {response.status_code}")
    real = response.json() if response.status_code == 201 else {}
    if real:
        print(f"      real: prob={real['fake_probability']} status={real['status']}")
        check(real["status"] == "REAL", f"real sample classified {real['status']}")
        check(real["fake_probability"] < 0.4, "real sample probability < 0.4")
        check(len(real["frame_scores"]) > 0, "real sample has frame scores")

    # --- happy path: fake sample -------------------------------------------
    response = upload(client, args.fake)
    check(response.status_code == 201, f"POST /analyze (fake) -> {response.status_code}")
    fake = response.json() if response.status_code == 201 else {}
    if fake:
        print(f"      fake: prob={fake['fake_probability']} status={fake['status']}")
        check(fake["status"] == "HIGH_RISK", f"fake sample classified {fake['status']}")
        check(fake["fake_probability"] > 0.7, "fake sample probability > 0.7")
        check(
            fake["suspicious_start"] is not None and fake["suspicious_end"] is not None,
            "fake sample has a suspicious region",
        )

    # --- GET /analysis/{id} -------------------------------------------------
    if fake:
        response = client.get(f"/analysis/{fake['id']}")
        check(response.status_code == 200, f"GET /analysis/{fake['id']} -> {response.status_code}")
        check(response.json() == fake, "GET by id matches the upload response")

    # --- GET /analysis/{id}/video -------------------------------------------
    if fake:
        check(fake["has_video"], "upload response reports a stored video")
        response = client.get(f"/analysis/{fake['id']}/video")
        check(
            response.status_code == 200,
            f"GET /analysis/{fake['id']}/video -> {response.status_code}",
        )
        check(
            response.headers.get("content-type") == "video/mp4",
            f"video content-type = {response.headers.get('content-type')}",
        )
        # Range support is what lets the player seek without a full download.
        response = client.get(
            f"/analysis/{fake['id']}/video", headers={"Range": "bytes=0-99"}
        )
        check(response.status_code == 206, f"ranged video GET -> {response.status_code} (want 206)")
        check(len(response.content) == 100, f"ranged video GET returned {len(response.content)} bytes")

    # --- GET /history -------------------------------------------------------
    response = client.get("/history")
    check(response.status_code == 200, f"GET /history -> {response.status_code}")
    rows = response.json()
    check(len(rows) >= 2, f"history has both rows ({len(rows)} total)")
    if len(rows) >= 2 and fake and real:
        check(rows[0]["id"] == fake["id"], "history is newest first")
        check(
            {real["id"], fake["id"]} <= {row["id"] for row in rows},
            "history contains both uploads",
        )

    # --- T8 error cases -----------------------------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        bad = Path(tmp) / "notes.txt"
        bad.write_text("not a video")
        response = upload(client, bad)
        check(response.status_code == 400, f"bad extension -> {response.status_code} (want 400)")

        empty = Path(tmp) / "empty.mp4"
        empty.touch()
        response = upload(client, empty)
        check(response.status_code == 400, f"empty file -> {response.status_code} (want 400)")

        garbage = Path(tmp) / "garbage.mp4"
        garbage.write_bytes(b"\x00" * 4096)
        response = upload(client, garbage)
        check(response.status_code == 422, f"unreadable video -> {response.status_code} (want 422)")

        faceless = _write_faceless_video(Path(tmp) / "faceless.mp4")
        response = upload(client, faceless)
        check(response.status_code == 422, f"no faces -> {response.status_code} (want 422)")

        oversized = Path(tmp) / "big.mp4"
        with open(oversized, "wb") as handle:
            handle.write(b"\x00" * (201 * 1024 * 1024))
        response = upload(client, oversized)
        check(response.status_code == 413, f"oversized file -> {response.status_code} (want 413)")

    response = client.get("/analysis/999999")
    check(response.status_code == 404, f"unknown id -> {response.status_code} (want 404)")

    response = client.get("/analysis/999999/video")
    check(response.status_code == 404, f"video for unknown id -> {response.status_code} (want 404)")

    print()
    if failures:
        print(f"SMOKE TEST FAILED ({len(failures)} checks)")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
