from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

ML_DIR = Path(__file__).resolve().parent
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

import protocol  # noqa: E402

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
FLAT_SET_LIMIT = 900

FFPP_METHODS = (
    "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures",
    "FaceShifter", "DeepFakeDetection",
)


def stable_split(identity: str, val: float = 0.15, test: float = 0.15) -> str:
    digest = hashlib.sha1(identity.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:4], "big") / 0xFFFFFFFF
    if bucket < test:
        return "test"
    if bucket < test + val:
        return "val"
    return "train"


def identity_of(video: str, dataset: str) -> str:
    if dataset == "cdf":
        match = re.match(r"(id\d+)", video)
        return f"cdf_{match.group(1)}" if match else f"cdf_yt_{video}"
    if dataset in {"ffpp", "df40"}:
        head = video.split("_")[0]
        return f"ff_{head}" if head.isdigit() else f"{dataset}_{video}"
    return f"{dataset}_{video}"


def subsample(names: list[str], limit: int) -> list[str]:
    names = sorted(names)
    if limit <= 0 or len(names) <= limit:
        return names
    step = len(names) / limit
    return [names[int(i * step)] for i in range(limit)]


class Writer:

    def __init__(self, out: Path):
        self.out = out
        self.rows: list[dict] = []
        self._members: set[str] = set()

    def add_zip(
        self,
        zip_path: Path,
        plan: dict[tuple[str, str, str], list[str]],
        archive: zipfile.ZipFile,
        group_override: dict[tuple[str, str, str], str] | None = None,
    ) -> None:
        group_override = group_override or {}
        self._members = set(archive.namelist())
        progress = tqdm(
            plan.items(), desc=zip_path.stem, leave=False,
            disable=not sys.stderr.isatty(), mininterval=5.0,
        )
        for (dataset, method, video), members in progress:
            identity = identity_of(video, dataset)
            split = stable_split(identity)
            group = group_override.get(
                (dataset, method, video), protocol.group_for(method, dataset)
            )
            target_dir = self.out / dataset / method / video
            target_dir.mkdir(parents=True, exist_ok=True)
            label = "real" if method == "real" else "fake"
            landmark_dir = self.out / dataset / "_landmarks" / video
            for member in members:
                name = Path(member).name
                destination = target_dir / name
                if not destination.exists():
                    with archive.open(member) as source:
                        destination.write_bytes(source.read())
                landmark_rel = ""
                if label == "real":
                    candidate = re.sub(r"/frames/", "/landmarks/", member)
                    candidate = str(Path(candidate).with_suffix(".npy"))
                    if candidate in self._members:
                        landmark_path = landmark_dir / f"{Path(name).stem}.npy"
                        if not landmark_path.exists():
                            landmark_dir.mkdir(parents=True, exist_ok=True)
                            with archive.open(candidate) as source:
                                landmark_path.write_bytes(source.read())
                        landmark_rel = str(landmark_path.relative_to(self.out))
                self.rows.append(
                    {
                        "path": str(destination.relative_to(self.out)),
                        "label": label,
                        "dataset": dataset,
                        "method": method,
                        "identity": identity,
                        "split": split,
                        "group": group,
                        "landmark": landmark_rel,
                    }
                )

    FIELDS = ["path", "label", "dataset", "method", "identity", "split",
              "group", "landmark"]

    def load_existing(self) -> int:
        path = self.out / "manifest.csv"
        if not path.is_file():
            return 0
        with open(path, newline="", encoding="utf-8") as handle:
            existing = list(csv.DictReader(handle))
        self.rows = existing + self.rows
        return len(existing)

    def write_manifest(self) -> Path:
        path = self.out / "manifest.csv"
        by_path = {row["path"]: row for row in self.rows}
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.FIELDS)
            writer.writeheader()
            writer.writerows(by_path.values())
        self.rows = list(by_path.values())
        return path


def usable_images(archive: zipfile.ZipFile) -> set[str]:
    return {
        info.filename for info in archive.infolist()
        if not info.filename.endswith("/")
        and info.file_size > 0
        and Path(info.filename).suffix.lower() in IMAGE_SUFFIXES
        and "/landmarks/" not in info.filename
        and "/masks/" not in info.filename
        and "__MACOSX" not in info.filename
    }


def group_members(archive: zipfile.ZipFile, pattern: re.Pattern) -> dict[str, list[str]]:
    videos: dict[str, list[str]] = defaultdict(list)
    usable = usable_images(archive)
    for name in archive.namelist():
        if name not in usable:
            continue
        match = pattern.search(name)
        if match:
            videos[match.group("video")].append(name)
    return videos


def do_celebdf(writer: Writer, zip_path: Path, per_video: int, real_per_video: int) -> None:
    with zipfile.ZipFile(zip_path) as archive:
        plan: dict[tuple[str, str, str], list[str]] = {}
        for folder, method, limit in (
            ("Celeb-real", "real", real_per_video),
            ("YouTube-real", "real", real_per_video),
            ("Celeb-synthesis", "cdf_synthesis", per_video // 2 or 1),
        ):
            pattern = re.compile(rf"^{folder}/frames/(?P<video>[^/]+)/[^/]+$")
            for video, members in group_members(archive, pattern).items():
                plan[("cdf", method, video)] = subsample(members, limit)
        writer.add_zip(zip_path, plan, archive)


def do_ffpp(writer: Writer, zip_path: Path, per_video: int) -> None:
    with zipfile.ZipFile(zip_path) as archive:
        plan: dict[tuple[str, str, str], list[str]] = {}
        real_pattern = re.compile(
            r"original_sequences/[^/]+/c23/frames/(?P<video>[^/]+)/[^/]+$"
        )
        for video, members in group_members(archive, real_pattern).items():
            plan[("ffpp", "real", video)] = subsample(members, per_video)
        for method in FFPP_METHODS:
            pattern = re.compile(
                rf"manipulated_sequences/{method}/c23/frames/(?P<video>[^/]+)/[^/]+$"
            )
            for video, members in group_members(archive, pattern).items():
                plan[("ffpp", method, video)] = subsample(members, per_video)
        writer.add_zip(zip_path, plan, archive)


def do_simple_bench(writer: Writer, zip_path: Path, dataset: str, per_video: int) -> None:
    with zipfile.ZipFile(zip_path) as archive:
        plan: dict[tuple[str, str, str], list[str]] = {}
        pattern = re.compile(r"(?P<label>[^/]*)/frames/(?P<video>[^/]+)/[^/]+$")
        usable = usable_images(archive)
        for name in archive.namelist():
            if name not in usable:
                continue
            match = pattern.search(name)
            if not match:
                continue
            lowered = name.lower()
            is_fake = any(
                key in lowered
                for key in ("fake", "manipulated", "synthesis", "method_a", "method_b")
            )
            method = f"{dataset}_fake" if is_fake else "real"
            key = (dataset, method, match.group("video"))
            plan.setdefault(key, []).append(name)
        plan = {key: subsample(members, per_video) for key, members in plan.items()}
        writer.add_zip(zip_path, plan, archive)


def do_df40(writer: Writer, zip_path: Path, per_video: int) -> None:
    method = zip_path.stem
    with zipfile.ZipFile(zip_path) as archive:
        usable = usable_images(archive)
        names = [n for n in archive.namelist() if n in usable]
        plan: dict[tuple[str, str, str], list[str]] = {}
        flat_keys: set[tuple[str, str, str]] = set()
        group_override: dict[tuple[str, str, str], str] = {}

        domain_pattern = re.compile(
            r"^[^/]+/(?P<domain>cdf|ff)/frames/(?P<video>[^/]+)/[^/]+$"
        )
        labelled_frames_pattern = re.compile(
            r"^[^/]+/(?P<label>real|fake)/frames/(?P<video>[^/]+)/[^/]+$"
        )
        frames_pattern = re.compile(r"^[^/]+/frames/(?P<video>[^/]+)/[^/]+$")
        domain_bucket_pattern = re.compile(
            r"^[^/]+/(?P<domain>cdf|ff)/(?P<bucket>[^/]+)/(?P<video>[^/]+)/[^/]+$"
        )
        domain_identity_pattern = re.compile(
            r"^[^/]+/(?P<domain>cdf|ff)/(?P<video>[^/]+)/[^/]+$"
        )
        paired_pattern = re.compile(
            r"^[^/]+/(?P<label>real|fake)(?:/(?:real|fake))?/(?:(?P<video>[^/]+)/)?[^/]+$"
        )
        identity_pattern = re.compile(r"^[^/]+/(?P<video>[^/]+)/[^/]+$")

        unmatched = 0
        unmatched_examples: list[str] = []

        for name in names:
            match = domain_pattern.match(name)
            if match:
                video = f"{match.group('domain')}_{match.group('video')}"
                plan.setdefault(("df40", method, video), []).append(name)
                continue

            match = labelled_frames_pattern.match(name)
            if match:
                label = match.group("label")
                video = match.group("video")
                if label == "fake":
                    key = ("df40", method, video)
                else:
                    key = ("df40", "real", f"{method}_real_{video}")
                    group_override[key] = protocol.group_for(method, "df40")
                plan.setdefault(key, []).append(name)
                continue

            match = frames_pattern.match(name)
            if match:
                plan.setdefault(
                    ("df40", method, f"ff_{match.group('video')}"), []
                ).append(name)
                continue

            match = domain_bucket_pattern.match(name)
            if match:
                video = (f"{match.group('domain')}_{match.group('bucket')}"
                         f"_{match.group('video')}")
                plan.setdefault(("df40", method, video), []).append(name)
                continue

            match = domain_identity_pattern.match(name)
            if match:
                video = f"{match.group('domain')}_{match.group('video')}"
                plan.setdefault(("df40", method, video), []).append(name)
                continue

            match = paired_pattern.match(name)
            if match:
                label = match.group("label")
                row_method = method if label == "fake" else "real"
                video = match.group("video") or f"{method}_{label}"
                key = ("df40", row_method, f"{method}_{label}_{video}")
                plan.setdefault(key, []).append(name)
                flat_keys.add(key)
                if row_method == "real":
                    group_override[key] = protocol.group_for(method, "df40")
                continue

            match = identity_pattern.match(name)
            if match:
                plan.setdefault(
                    ("df40", method, match.group("video")), []
                ).append(name)
                continue

            unmatched += 1
            if len(unmatched_examples) < 3:
                unmatched_examples.append(name)

        if unmatched:
            print(f"  WARNING: {unmatched}/{len(names)} members in "
                  f"{zip_path.name} match no known layout, e.g. "
                  f"{unmatched_examples}", flush=True)

        plan = {
            key: subsample(members, FLAT_SET_LIMIT if key in flat_keys else per_video)
            for key, members in plan.items()
        }
        writer.add_zip(zip_path, plan, archive, group_override)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dl", default="data/dl")
    parser.add_argument("--out", default="data/processed_v2")
    parser.add_argument("--frames-per-video", type=int, default=8)
    parser.add_argument("--real-frames-per-video", type=int, default=16)
    parser.add_argument("--only", nargs="*", default=None,
                        help="restrict to these zip stems")
    parser.add_argument("--merge", action="store_true",
                        help="keep the rows already in manifest.csv; required "
                             "with --only, which otherwise writes a manifest "
                             "holding nothing but the zips it rebuilt")
    args = parser.parse_args()

    dl = Path(args.dl)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    writer = Writer(out)
    if args.merge:
        print(f"merging into {writer.load_existing()} existing manifest rows",
              flush=True)

    bench = dl / "Ella4839__DeepfakeBench"
    jobs: list[tuple[str, Path]] = []
    if (bench / "Celeb-DF-v2.zip").is_file():
        jobs.append(("cdf", bench / "Celeb-DF-v2.zip"))
    if (bench / "FaceForensics++.zip").is_file():
        jobs.append(("ffpp", bench / "FaceForensics++.zip"))
    if (bench / "DFDCP.zip").is_file():
        jobs.append(("dfdcp", bench / "DFDCP.zip"))
    if (bench / "UADFV.zip").is_file():
        jobs.append(("uadfv", bench / "UADFV.zip"))
    for repo in ("ManhQuangAI__DF40_train", "ManhQuangAI__df-40-test-full"):
        for zip_path in sorted((dl / repo).glob("*.zip")):
            jobs.append(("df40", zip_path))

    if args.only:
        jobs = [job for job in jobs if job[1].stem in args.only]

    for kind, zip_path in jobs:
        print(f"--- {kind}: {zip_path.name}", flush=True)
        try:
            if kind == "cdf":
                do_celebdf(writer, zip_path, args.frames_per_video,
                           args.real_frames_per_video)
            elif kind == "ffpp":
                do_ffpp(writer, zip_path, args.frames_per_video)
            elif kind == "df40":
                do_df40(writer, zip_path, args.frames_per_video)
            else:
                do_simple_bench(writer, zip_path, kind, args.frames_per_video)
        except zipfile.BadZipFile as exc:
            print(f"  skipping {zip_path.name}: {exc}", flush=True)

    path = writer.write_manifest()
    print(f"\nwrote {len(writer.rows)} rows -> {path}")

    import pandas as pd

    frame = pd.read_csv(path)
    print(frame.groupby(["dataset", "group", "label"]).size().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
