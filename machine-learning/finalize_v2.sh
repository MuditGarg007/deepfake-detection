#!/usr/bin/env bash

set -u
cd "$(dirname "$0")/.." || exit 1

PY=.venv/bin/python
DATA=data/processed_v2
LOGS=logs/experiments
RUNS=machine-learning/runs
mkdir -p "$LOGS"

EVAL_COMMON=(--data "$DATA" --split test --amp --max-per-method 800
             --max-real 4000 --heldout-all-splits)
HEADLINE=(--corruptions clean jpeg_q40 jpeg_q10 downscale_0.25)

say () { echo "=== [$(date +%H:%M:%S)] $*"; }

say "re-evaluating all checkpoints on the current manifest"
for ckpt in machine-learning/checkpoints/*/; do
  ckpt="${ckpt%/}"
  name=$(basename "$ckpt")
  [ -f "$ckpt/config.json" ] || continue
  [ -f "$ckpt/best.pth" ] || continue
  say "eval $name"
  $PY machine-learning/evaluate_v2.py --checkpoint "$ckpt" \
      "${EVAL_COMMON[@]}" "${HEADLINE[@]}" \
      > "$LOGS/final_eval_${name}.log" 2>&1 \
    || { echo "  FAILED — see $LOGS/final_eval_${name}.log"; tail -3 "$LOGS/final_eval_${name}.log"; }
  grep -E "^  clean" "$LOGS/final_eval_${name}.log" || true
done

say "ranking"
$PY machine-learning/rank_runs.py | tee "$RUNS/ranking.txt"
mapfile -t FINALISTS < <($PY machine-learning/rank_runs.py --top 3)
if [ "${#FINALISTS[@]}" -eq 0 ]; then
  echo "no comparable evaluations — stopping"; exit 1
fi
WINNER="${FINALISTS[0]}"
say "winner: $(basename "$WINNER")"

for ckpt in "${FINALISTS[@]:0:2}"; do
  name=$(basename "$ckpt")
  say "full corruption sweep: $name"
  $PY machine-learning/evaluate_v2.py --checkpoint "$ckpt" "${EVAL_COMMON[@]}" \
      --out "$RUNS/corruption_${name}" \
      > "$LOGS/corruption_${name}.log" 2>&1 \
    || { echo "  FAILED — see $LOGS/corruption_${name}.log"; tail -3 "$LOGS/corruption_${name}.log"; }
  grep -E "^  [a-z]" "$LOGS/corruption_${name}.log" || true
done

say "score-distribution diagnosis: $(basename "$WINNER")"
$PY machine-learning/diagnose_v2.py --checkpoint "$WINNER" "${EVAL_COMMON[@]}" \
    > "$LOGS/diagnose_$(basename "$WINNER").log" 2>&1 \
  || tail -3 "$LOGS/diagnose_$(basename "$WINNER").log"

say "calibrating $(basename "$WINNER")"
$PY machine-learning/calibrate_v2.py --checkpoint "$WINNER" --data "$DATA" \
    --split val --amp --write \
    > "$LOGS/calibrate_$(basename "$WINNER").log" 2>&1 \
  || tail -5 "$LOGS/calibrate_$(basename "$WINNER").log"
grep -E "^chosen|^  (clean|jpeg|downscale)" "$LOGS/calibrate_$(basename "$WINNER").log" || true

say "comparison table"
$PY machine-learning/report_v2.py > "$LOGS/report.log" 2>&1 || tail -5 "$LOGS/report.log"
tail -20 "$LOGS/report.log"

if [ "${#FINALISTS[@]}" -ge 2 ]; then
  say "ensemble check"
  $PY machine-learning/report_v2.py --amp --heldout-all-splits \
      --corruptions clean jpeg_q40 \
      --ensemble "${FINALISTS[@]}" \
      > "$LOGS/ensemble.log" 2>&1 || tail -5 "$LOGS/ensemble.log"
  grep -E "held-out macro" "$LOGS/ensemble.log" || true
fi

say "finalize done"
