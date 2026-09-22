"""Rank the finished evaluations by held-out AUC and print checkpoint paths.

The sweep leaves one ``runs/eval_*/summary.json`` per configuration. Choosing
the finalists by hand means reading eleven of them; this prints the ranking, or
just the top ``--top`` checkpoint directories for a script to consume.

Ranking is on clean held-out macro AUC — mean per-method AUC over the
manipulations and corpora excluded from training — because that is the only
column that says anything about a manipulation nobody has seen yet.

The method count is printed beside it and gates ``--top``, because that average
is only comparable between runs that averaged over the same thing. The v1
baseline's stored evaluation covers a single manipulation (Celeb-DF synthesis)
and its 0.84 sits above two v2 runs' 31-method averages while measuring
something else entirely.
"""

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
