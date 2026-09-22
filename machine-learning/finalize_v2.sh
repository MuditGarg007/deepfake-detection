#!/usr/bin/env bash
# Everything that has to happen after the sweep, in one unattended pass.
#
#   bash machine-learning/finalize_v2.sh
#
# The sweep evaluates each run as it finishes, which means runs trained before
# a corpus change were scored against a different set of manipulations. So the
# first thing here is to re-score every checkpoint — including the v1 baseline,
# which has only ever been evaluated on Celeb-DF — against the current manifest
# with the same flags. Only then is the comparison table a comparison.
#
# After that: the full corruption sweep and a score-distribution diagnosis on
# the finalists, a calibrated operating point written into the winner, and the
# ensemble check.

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

# --- 1. every checkpoint, same manifest, same flags ---------------------------
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

# --- 2. ranking ---------------------------------------------------------------
say "ranking"
$PY machine-learning/rank_runs.py | tee "$RUNS/ranking.txt"
mapfile -t FINALISTS < <($PY machine-learning/rank_runs.py --top 3)
if [ "${#FINALISTS[@]}" -eq 0 ]; then
  echo "no comparable evaluations — stopping"; exit 1
fi
WINNER="${FINALISTS[0]}"
say "winner: $(basename "$WINNER")"

# --- 3. the full twelve corruptions, finalists only ---------------------------
for ckpt in "${FINALISTS[@]:0:2}"; do
  name=$(basename "$ckpt")
  say "full corruption sweep: $name"
  $PY machine-learning/evaluate_v2.py --checkpoint "$ckpt" "${EVAL_COMMON[@]}" \
      --out "$RUNS/corruption_${name}" \
      > "$LOGS/corruption_${name}.log" 2>&1 \
    || { echo "  FAILED — see $LOGS/corruption_${name}.log"; tail -3 "$LOGS/corruption_${name}.log"; }
  grep -E "^  [a-z]" "$LOGS/corruption_${name}.log" || true
done

# --- 4. where the winner's scores sit, per method and per family --------------
say "score-distribution diagnosis: $(basename "$WINNER")"
$PY machine-learning/diagnose_v2.py --checkpoint "$WINNER" "${EVAL_COMMON[@]}" \
    > "$LOGS/diagnose_$(basename "$WINNER").log" 2>&1 \
  || tail -3 "$LOGS/diagnose_$(basename "$WINNER").log"

# --- 5. operating point -------------------------------------------------------
# On val, never test, and under a mixture of clean and degraded conditions —
# the failure being fixed is a threshold chosen on pristine PNG and applied to
# codec output.
say "calibrating $(basename "$WINNER")"
$PY machine-learning/calibrate_v2.py --checkpoint "$WINNER" --data "$DATA" \
    --split val --amp --write \
    > "$LOGS/calibrate_$(basename "$WINNER").log" 2>&1 \
  || tail -5 "$LOGS/calibrate_$(basename "$WINNER").log"
grep -E "^chosen|^  (clean|jpeg|downscale)" "$LOGS/calibrate_$(basename "$WINNER").log" || true

# --- 6. comparison table and ensemble ----------------------------------------
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
