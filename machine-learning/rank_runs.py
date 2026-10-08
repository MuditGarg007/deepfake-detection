from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="machine-learning/runs")
    parser.add_argument("--top", type=int, default=0,
                        help="print only this many checkpoint paths, one per line")
    parser.add_argument("--metric", default="heldout_macro_auc")
    parser.add_argument("--min-methods", type=int, default=10,
                        help="ignore evaluations that averaged over fewer "
                             "held-out manipulations than this when picking "
                             "--top; their average is not the same quantity")
    args = parser.parse_args()

    rows = []
    for path in sorted(Path(args.runs).glob("eval_*/summary.json")):
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        clean = payload.get("report", {}).get("clean", {}).get("overall", {})
        value = clean.get(args.metric)
        if value is None:
            continue
        methods = 0
        per_method = path.parent / "per_method.csv"
        if per_method.is_file():
            with open(per_method, newline="", encoding="utf-8") as handle:
                methods = len({
                    row["method"] for row in csv.DictReader(handle)
                    if row["corruption"] == "clean" and row["group"] == "heldout"
                })
        rows.append((value, payload["checkpoint"], path.parent.name,
                     clean.get("real_fpr@0.5", float("nan")), methods))
    rows.sort(reverse=True)

    if args.top:
        eligible = [r for r in rows if r[4] >= args.min_methods]
        for _, checkpoint, _, _, _ in eligible[:args.top]:
            print(checkpoint)
        return 0

    for value, _, name, fpr, methods in rows:
        flag = "" if methods >= args.min_methods else "   <- not comparable"
        print(f"{value:.4f}  fpr {fpr:.3f}  {methods:>2} held-out methods  "
              f"{name.removeprefix('eval_')}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
