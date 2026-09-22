#!/usr/bin/env bash
# v2 experiment sweep.
#
# Each run trains one configuration and evaluates it on the held-out test
# split. Ordered so the cheap, high-information runs land first: the data/
# augmentation effect is isolated on the v1 architecture before any of the
# expensive foundation-model runs start.
#
#   bash machine-learning/run_experiments.sh [stage]
#
# stage: "core" (the three runs the conclusion depends on), "ablations" (the
# one-factor-at-a-time ViT-B runs), "robust" (the wide-augmentation retrains
# that test the JPEG-q10 calibration collapse), "large" (the remaining big
# backbones), or "all" (all four, in that order).
#
# "core" runs first and deliberately duplicates nothing: an interrupted sweep
# should still leave the control, the main recipe and the best candidate on
# disk, because those three are what any conclusion rests on.

set -u
cd "$(dirname "$0")/.." || exit 1

PY=.venv/bin/python
DATA=data/processed_v2
LOGS=logs/experiments
mkdir -p "$LOGS"

SAMPLES=40000
EPOCHS=5

run () {
  local name="$1"; shift
  local log="$LOGS/${name}.log"
  if grep -q "^checkpoint: " "$log" 2>/dev/null; then
    echo "[skip] $name already complete"
    return 0
  fi
  echo "=== [$(date +%H:%M:%S)] $name"
  $PY machine-learning/train_v2.py --data "$DATA" --amp \
      --epochs "$EPOCHS" --samples-per-epoch "$SAMPLES" \
      --num-workers 6 --tag "$name" "$@" > "$log" 2>&1
  local status=$?
  if [ $status -ne 0 ]; then
    echo "  FAILED ($status) — see $log"
    tail -5 "$log"
    return 1
  fi
  local ckpt
  ckpt=$(grep "^checkpoint: " "$log" | tail -1 | cut -d' ' -f2)
  echo "  trained -> $ckpt"
  grep "^best epoch" "$log"

  # Four corruptions, not the full twelve: the whole sweep costs more in
  # evaluation than in training otherwise. The finalists get the full sweep.
  #
  # --heldout-all-splits because five held-out image-synthesis sets are flat
  # piles with no video structure, so the identity hash puts each of them
  # entirely into one split — and for all five that split is not "test". They
  # were absent from every per-method table until this flag existed.
  $PY machine-learning/evaluate_v2.py --checkpoint "$ckpt" --data "$DATA" \
      --split test --amp --max-per-method 800 --max-real 4000 \
      --heldout-all-splits \
      --corruptions clean jpeg_q40 jpeg_q10 downscale_0.25 \
      > "$LOGS/${name}.eval.log" 2>&1
  grep -E "^  (clean|jpeg_q40|jpeg_q10|downscale_0.25)" "$LOGS/${name}.eval.log"
}

stage="${1:-ablations}"

if [ "$stage" = "core" ] || [ "$stage" = "all" ]; then
  # v1 architecture, v2 data + augmentation. Isolates how much of the gain is
  # the training corpus rather than the backbone.
  run effnet_data_only --backbone efficientnet_b0 --tune full --head mlp \
      --augment heavy --sbi-prob 0.25 --lr 1e-4 --batch-size 48 \
      --feature-noise 0 --feature-mixup 0

  # The main v2 recipe at small scale.
  run clipb_full --backbone clip_vit_b16 --tune ln --head hypersphere \
      --augment heavy --sbi-prob 0.25 --lr 1e-3 --batch-size 48

  # The expected best single model.
  run clipl_full --backbone clip_vit_l14 --tune ln --head hypersphere \
      --augment heavy --sbi-prob 0.25 --lr 1e-3 --batch-size 24
fi

if [ "$stage" = "ablations" ] || [ "$stage" = "all" ]; then
  # Ablations against clipb_full, one factor at a time.
  run clipb_no_sbi --backbone clip_vit_b16 --tune ln --head hypersphere \
      --augment heavy --sbi-prob 0.0 --lr 1e-3 --batch-size 48
  run clipb_light_aug --backbone clip_vit_b16 --tune ln --head hypersphere \
      --augment light --sbi-prob 0.25 --lr 1e-3 --batch-size 48
  run clipb_linear_probe --backbone clip_vit_b16 --tune head --head hypersphere \
      --augment heavy --sbi-prob 0.25 --lr 3e-3 --batch-size 48
  run clipb_mlp_head --backbone clip_vit_b16 --tune ln --head mlp \
      --augment heavy --sbi-prob 0.25 --lr 1e-3 --batch-size 48
  run clipb_no_latent --backbone clip_vit_b16 --tune ln --head hypersphere \
      --augment heavy --sbi-prob 0.25 --lr 1e-3 --batch-size 48 \
      --feature-noise 0 --feature-mixup 0
fi

if [ "$stage" = "robust" ] || [ "$stage" = "all" ]; then
  # Same recipe as clipb_full/clipl_full, with the augmentation floor pushed
  # below the worst quality expected at inference. "heavy" tops out at JPEG q30
  # and the measured failure is at q10: AUC degrades gracefully there but the
  # real false-positive rate blows out from 0.086 to 0.493, which is a
  # calibration failure, not a discrimination one. If that is purely a
  # train/test range gap, heavy_wide closes it at no cost on clean data.
  run clipb_heavy_wide --backbone clip_vit_b16 --tune ln --head hypersphere \
      --augment heavy_wide --sbi-prob 0.25 --lr 1e-3 --batch-size 48
  run clipl_heavy_wide --backbone clip_vit_l14 --tune ln --head hypersphere \
      --augment heavy_wide --sbi-prob 0.25 --lr 1e-3 --batch-size 24
fi

if [ "$stage" = "large" ] || [ "$stage" = "all" ]; then
  run dinov3l_full --backbone dinov3_vit_l16 --tune ln --head hypersphere \
      --augment heavy --sbi-prob 0.25 --lr 1e-3 --batch-size 24
  run convnext_full --backbone convnext_base_clip --tune ln --head hypersphere \
      --augment heavy --sbi-prob 0.25 --lr 1e-3 --batch-size 32
fi

echo "=== sweep finished at $(date +%H:%M:%S)"
