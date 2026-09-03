"""T9 — build ``machine-learning/runs/comparison.md`` from evaluated models.

Usage::

    python machine-learning/compare.py \
        --checkpoints machine-learning/checkpoints/efficientnet_b0_* \
                      machine-learning/checkpoints/xception_*
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ML_DIR = Path(__file__).resolve().parent

# ROC-AUC differences smaller than this are treated as a tie and broken on F1.
AUC_TIE_MARGIN = 0.001


def load_metrics(checkpoint_dir: Path, runs_root: Path) -> dict:
    """Read the metrics.json written by evaluate.py for this checkpoint."""
    with open(checkpoint_dir / "config.json", encoding="utf-8") as handle:
        config = json.load(handle)
    run_name = config.get("run_name", checkpoint_dir.name)
    metrics_path = runs_root / run_name / "metrics.json"
    if not metrics_path.is_file():
        raise SystemExit(f"{metrics_path} not found — run evaluate.py for {run_name}")
    with open(metrics_path, encoding="utf-8") as handle:
        metrics = json.load(handle)
    metrics["config"] = config
    return metrics


def render(entries: list[dict]) -> str:
    """Render the comparison table plus a conclusion paragraph."""
    header = (
        "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Infer ms/img |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    rows = "".join(
        "| {model} | {acc:.4f} | {prec:.4f} | {rec:.4f} | {f1:.4f} | {auc:.4f} | {ms:.2f} |\n".format(
            model=e["model"],
            acc=e["accuracy"],
            prec=e["precision"],
            rec=e["recall"],
            f1=e["f1"],
            auc=e["roc_auc"],
            ms=e["inference_ms_per_image"],
        )
        for e in entries
    )

    best_auc = max(entries, key=lambda e: e["roc_auc"])
    best_f1 = max(entries, key=lambda e: e["f1"])
    fastest = min(entries, key=lambda e: e["inference_ms_per_image"])
    sample = entries[0]

    # ROC-AUC is the primary metric (threshold-free; Phase 2 picks its own
    # cutoffs from thresholds.csv). But when the AUC spread is within noise,
    # ranking on it alone picks a winner on nothing, so fall back to F1.
    auc_spread = best_auc["roc_auc"] - min(e["roc_auc"] for e in entries)
    tie = len(entries) > 1 and auc_spread < AUC_TIE_MARGIN
    winner = best_f1 if tie else best_auc

    conclusion = (
        f"**{best_auc['model']}** achieves the highest ROC-AUC "
        f"({best_auc['roc_auc']:.4f}); **{best_f1['model']}** has the best F1 "
        f"({best_f1['f1']:.4f}), and **{fastest['model']}** is the fastest at "
        f"{fastest['inference_ms_per_image']:.2f} ms/image on "
        f"{sample.get('gpu') or sample.get('device')}. "
    )
    if tie:
        conclusion += (
            f"The ROC-AUC spread across the models is {auc_spread:.4f}, below the "
            f"{AUC_TIE_MARGIN} margin this report treats as a tie, so the decision "
            f"falls to F1 at the 0.5 threshold: **{winner['model']}** is the "
            f"checkpoint the backend should load."
        )
    else:
        conclusion += (
            "ROC-AUC is the selection metric for this project (it is threshold-free, "
            "and Phase 2 tunes its own risk cutoffs from `thresholds.csv`), so "
            f"**{winner['model']}** is the checkpoint the backend should load."
        )

    details = "".join(
        f"- **{e['model']}** — checkpoint `{e['checkpoint']}`, "
        f"best epoch {e['config'].get('best_epoch')}, "
        f"{e['num_samples']} test crops, "
        f"confusion matrix {e['confusion_matrix']}\n"
        for e in entries
    )

    return (
        "# Model comparison — Phase 1\n\n"
        f"Test split: `{sample['split']}` "
        f"({sample['num_samples']} face crops), decision threshold "
        f"{sample['threshold']}.\n\n"
        f"{header}{rows}\n"
        "## Conclusion\n\n"
        f"{conclusion}\n\n"
        "## Run details\n\n"
        f"{details}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--runs", default=str(ML_DIR / "runs"))
    parser.add_argument("--out", default=str(ML_DIR / "runs" / "comparison.md"))
    args = parser.parse_args()

    runs_root = Path(args.runs)
    entries = [load_metrics(Path(c), runs_root) for c in args.checkpoints]
    text = render(entries)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nwritten -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
