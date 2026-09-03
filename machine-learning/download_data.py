"""T2 — fetch the FaceForensics++ c23 videos used for training.

The official FaceForensics++ distribution is access-gated behind a request form,
so this project pulls the same c23 videos from the public Hugging Face mirror
``bitmind/FaceForensicsC23`` (7,000 manipulated + 1,000 real MP4s in one zip).

Only the two classes the classifier needs are extracted:

* ``Real``      -> label ``real``
* ``Deepfakes`` -> label ``fake``

Usage::

    python machine-learning/download_data.py --out data/raw

The zip is ~18 GB; the extracted subset is ~5 GB. Pass ``--keep-zip`` to leave
the archive on disk (it is deleted after a successful extraction by default).

The download is a resumable parallel range fetch: chunk progress is recorded in
a ``<zip>.progress.json`` sidecar, so re-running the command after an
interruption only fetches the chunks that are still missing.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

REPO_ID = "bitmind/FaceForensicsC23"
ZIP_NAME = "FaceForensics++_C23.zip"
URL = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/FaceForensics%2B%2B_C23.zip"
# Directory names inside the archive -> local class directory name. The mirror
# lays the archive out as ``FaceForensics++_C23/real/*.mp4`` and
# ``FaceForensics++_C23/fake/<method>/*.mp4``; only the Deepfakes method is used.
WANTED = {"real": "Real", "Deepfakes": "Deepfakes"}

CHUNK_BYTES = 64 * 1024 * 1024
MAX_RETRIES = 6


def human(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num_bytes) < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} PB"


def remote_size(url: str) -> int:
    """Total size of the file behind ``url`` (follows the HF 302 to the CDN)."""
    response = requests.head(url, allow_redirects=True, timeout=60)
    response.raise_for_status()
    size = response.headers.get("Content-Length")
    if size is None:
        raise SystemExit(f"server did not report a Content-Length for {url}")
    if response.headers.get("Accept-Ranges") != "bytes":
        raise SystemExit("server does not support range requests")
    return int(size)


def fetch_chunk(url: str, path: Path, index: int, start: int, end: int) -> int:
    """Download bytes ``[start, end]`` into ``path`` at that offset.

    The HF CDN hands out short-lived signed URLs, so every attempt re-resolves
    the redirect from the stable ``huggingface.co`` URL instead of caching one.
    """
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            headers = {"Range": f"bytes={start}-{end}"}
            with requests.get(url, headers=headers, stream=True, timeout=120) as response:
                response.raise_for_status()
                offset = start
                handle = os.open(path, os.O_WRONLY)
                try:
                    for block in response.iter_content(chunk_size=1 << 20):
                        if block:
                            os.pwrite(handle, block, offset)
                            offset += len(block)
                finally:
                    os.close(handle)
            if offset != end + 1:
                raise OSError(f"short read: got {offset - start} of {end - start + 1} bytes")
            return index
        except Exception as exc:  # transient network/CDN failure -> back off
            last_error = exc
            time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"chunk {index} failed after {MAX_RETRIES} attempts: {last_error}")


def download_zip(dest_dir: Path, workers: int) -> Path:
    """Resumable parallel download of the dataset archive into ``dest_dir``."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / ZIP_NAME
    progress_path = zip_path.with_suffix(".zip.progress.json")

    total = remote_size(URL)
    chunks = [
        (i, start, min(start + CHUNK_BYTES, total) - 1)
        for i, start in enumerate(range(0, total, CHUNK_BYTES))
    ]

    state = {"total": total, "chunk_bytes": CHUNK_BYTES, "done": []}
    if progress_path.is_file() and zip_path.is_file():
        saved = json.loads(progress_path.read_text())
        if saved.get("total") == total and saved.get("chunk_bytes") == CHUNK_BYTES:
            state = saved
    done = set(state["done"])

    if zip_path.is_file() and zip_path.stat().st_size == total and len(done) == len(chunks):
        print(f"{zip_path} already complete ({human(total)})")
        return zip_path

    # Preallocate so workers can write their slice at any offset.
    with open(zip_path, "ab"):
        pass
    os.truncate(zip_path, total)

    pending = [c for c in chunks if c[0] not in done]
    print(
        f"Downloading {human(total)} in {len(chunks)} chunks "
        f"({len(pending)} remaining) with {workers} workers ...",
        flush=True,
    )

    lock = threading.Lock()
    started = time.time()
    resumed_bytes = len(done) * CHUNK_BYTES
    session_bytes = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(fetch_chunk, URL, zip_path, i, start, end): (i, start, end)
            for i, start, end in pending
        }
        for future in as_completed(futures):
            index = future.result()
            _, start, end = futures[future]
            with lock:
                done.add(index)
                state["done"] = sorted(done)
                progress_path.write_text(json.dumps(state))
                session_bytes += end - start + 1
                rate = session_bytes / max(time.time() - started, 1e-9)
                print(
                    f"  {len(done)}/{len(chunks)} chunks  "
                    f"{human(resumed_bytes + session_bytes)}/{human(total)}  "
                    f"{human(rate)}/s",
                    flush=True,
                )

    progress_path.unlink(missing_ok=True)
    print(f"Downloaded -> {zip_path}")
    return zip_path


def extract_subset(zip_path: Path, out_dir: Path) -> dict[str, int]:
    """Extract only the Real/ and Deepfakes/ videos, flattened per class."""
    counts = {name: 0 for name in WANTED.values()}
    with zipfile.ZipFile(zip_path) as archive:
        members = [m for m in archive.namelist() if m.lower().endswith(".mp4")]
        if not members:
            raise SystemExit(f"no .mp4 entries found in {zip_path}")

        for member in members:
            parts = Path(member).parts
            cls = next((WANTED[p] for p in parts if p in WANTED), None)
            if cls is None:
                continue
            target = out_dir / cls / Path(member).name
            counts[cls] += 1
            if target.exists() and target.stat().st_size > 0:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst, length=1 << 20)
            if counts[cls] % 100 == 0:
                print(f"  {cls}: {counts[cls]} videos", flush=True)
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/raw", help="output directory")
    parser.add_argument("--workers", type=int, default=8, help="parallel range fetchers")
    parser.add_argument(
        "--zip",
        default=None,
        help="path to an already-downloaded archive (skips the download)",
    )
    parser.add_argument(
        "--keep-zip", action="store_true", help="keep the archive after extraction"
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    zip_path = Path(args.zip) if args.zip else download_zip(out_dir, args.workers)
    print(f"Extracting Real/ and Deepfakes/ from {zip_path} ...", flush=True)
    counts = extract_subset(zip_path, out_dir)

    for cls, n in counts.items():
        print(f"{cls}: {n} videos -> {out_dir / cls}")

    if not args.keep_zip and not args.zip:
        zip_path.unlink(missing_ok=True)
        print(f"Removed {zip_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
