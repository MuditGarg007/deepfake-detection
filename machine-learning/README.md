# Phase 1 — Deepfake face-crop classifier

Trains and compares two CNN classifiers that map a **224×224 face crop → fake
probability**. This is the model the backend loads in Phase 2
(`backend/services/detector.py` calls `inference.load_model` / `inference.predict`).

See [`docs/plan.md`](../docs/plan.md) for the whole system and
[`docs/ml-roadmap.md`](../docs/ml-roadmap.md) for the task-by-task plan this
implements.

---

## Contents

| File | Role |
|---|---|
| `download_data.py` | T2 — fetch FaceForensics++ c23 videos (real + Deepfakes) |
| `preprocessing.py` | T3 — videos → sampled frames → MTCNN face crops → `data/processed/` |
| `dataset.py` | T4 — `FaceDataset` + augmentations + dataloaders |
| `models.py` | T5 — EfficientNet-B0 / Xception builders with a binary head |
| `train.py` | T6 — fine-tuning loop (AMP, early stopping on val ROC-AUC) |
| `evaluate.py` | T7 — test metrics, ROC curve, confusion matrix, threshold sweep |
| `inference.py` | T8 — `load_model()` / `predict()`, the Phase 2 interface |
| `compare.py` | T9 — writes `runs/comparison.md` |

Outputs (both gitignored except `runs/comparison.md`):

```text
machine-learning/checkpoints/<model>_<timestamp>/{best.pth,config.json}
machine-learning/runs/<model>_<timestamp>/{log.csv,metrics.json,roc_curve.png,
                                           confusion_matrix.png,thresholds.csv}
machine-learning/runs/comparison.md
```

---

## Setup

Python **3.12** (torch has no 3.14 wheels yet). From the repo root:

```bash
uv venv --python 3.12 .venv          # or: python3.12 -m venv .venv
.venv/bin/python -m pip install -r machine-learning/requirements.txt
# facenet-pytorch's metadata pins torchvision<0.18, which conflicts with the
# CUDA-12/13 torch wheels; its MTCNN code works fine on current torchvision.
.venv/bin/python -m pip install --no-deps 'facenet-pytorch>=2.6'
```

GPU sanity check:

```bash
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## Dataset used

**FaceForensics++ c23**, taken from the public Hugging Face mirror
[`bitmind/FaceForensicsC23`](https://huggingface.co/datasets/bitmind/FaceForensicsC23)
rather than the access-gated official download, so the pipeline reproduces
without a request form. Only two of its seven folders are used:

| Class | Source folder | Videos |
|---|---|---|
| real | `Real/` | 1,000 |
| fake | `Deepfakes/` | 1,000 |

```bash
.venv/bin/python machine-learning/download_data.py --out data/raw
```

The archive is ~18 GB; the extracted two-class subset is ~5 GB and the archive
is deleted afterwards (`--keep-zip` to keep it).

---

## Reproduction

```bash
# 1. Face crops (identity-level 70/15/15 split — no frame or identity leakage)
.venv/bin/python machine-learning/preprocessing.py \
    --raw data/raw --out data/processed \
    --fps 5 --max-frames 50 --margin 20 --conf 0.95 --seed 42

# 2. Train both backbones
.venv/bin/python machine-learning/train.py --model efficientnet_b0 \
    --data data/processed --epochs 15 --batch-size 64 --lr 1e-4 --amp --patience 5
.venv/bin/python machine-learning/train.py --model xception \
    --data data/processed --epochs 15 --batch-size 32 --lr 1e-4 --amp --patience 5

# 3. Evaluate each checkpoint on the held-out test split
.venv/bin/python machine-learning/evaluate.py --amp \
    --checkpoint machine-learning/checkpoints/efficientnet_b0_<ts> --data data/processed
.venv/bin/python machine-learning/evaluate.py --amp \
    --checkpoint machine-learning/checkpoints/xception_<ts> --data data/processed

# 4. Inference smoke test (the interface Phase 2 consumes)
.venv/bin/python machine-learning/inference.py \
    --checkpoint machine-learning/checkpoints/<best>_<ts> --data data/processed --limit 100

# 5. Comparison report
.venv/bin/python machine-learning/compare.py --checkpoints \
    machine-learning/checkpoints/efficientnet_b0_<ts> \
    machine-learning/checkpoints/xception_<ts>
```

---

## Training setup

| Setting | Value |
|---|---|
| Split | 70/15/15 by FF++ source identity (`033.mp4` and `033_097.mp4` stay together) |
| Input | 224×224 RGB face crop, ImageNet normalization |
| Augmentation (train only) | h-flip 0.5, affine ±10° / ±5% translate, jitter 0.2/0.2 |
| Head | pooled features → 128 → ReLU → dropout 0.2 → 1 logit |
| Loss | `BCEWithLogitsLoss` (`pos_weight` from the train class ratio) |
| Optimizer | Adam, lr 1e-4, weight decay 1e-5 |
| Precision | AMP fp16 with `GradScaler`, grad-norm clip 1.0 |
| Early stopping | val ROC-AUC, patience 5 |
| Fine-tuning | full backbone (no frozen layers) |

---

## Results

Hardware: NVIDIA GeForce RTX 4060 Laptop GPU (8 GB), 16-core CPU, 15 GB RAM.
Dataset: 99,586 face crops from 2,000 videos — train 69,767 / val 14,887 /
test 14,932, near-perfectly balanced (49.98 % fake overall). MTCNN found a face
above confidence 0.95 in 99.83 % of sampled frames.

Test-split results at threshold 0.5 (full numbers in `runs/<run>/metrics.json`,
report in [`runs/comparison.md`](runs/comparison.md)):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Infer ms/img | Best epoch | Train time |
|---|---|---|---|---|---|---|---|---|
| EfficientNet-B0 | 0.9912 | 0.9887 | 0.9938 | 0.9912 | 0.9989 | 2.76 | 8 (early stop @13) | 24 min |
| Xception | 0.9858 | 0.9919 | 0.9796 | 0.9857 | 0.9990 | 2.08 | 6 (early stop @11) | 42 min |

Both clear the ≥ 0.90 ROC-AUC exit criterion by a wide margin. The AUC gap
between them is 0.0001 — noise — so **EfficientNet-B0 is the recommended
checkpoint**: it wins on accuracy and F1, is 5× smaller (4.2 M vs 21.1 M
parameters), and trains in half the time. Xception is the more conservative
model (higher precision, lower recall): it misses more fakes but raises fewer
false alarms on real video.

Total end-to-end wall time on this machine: ~1 h download, 12 min preprocessing,
66 min training both models.

### Threshold behaviour at the Phase 2 risk cutoffs

From `runs/efficientnet_b0_*/thresholds.csv`, for the plan's 40 % / 70 % bands:

| Threshold | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| 0.41 | 0.9880 | 0.9949 | 0.9915 | 0.9914 |
| 0.70 | 0.9905 | 0.9910 | 0.9908 | 0.9908 |

Frame-level performance is essentially flat across that whole band, so the
plan's default cutoffs need no retuning.

### Caveat

These numbers are for the **FF++ Deepfakes c23** manipulation only, which is
what Phase 1 scoped. The split is by source identity, so there is no identity or
frame leakage between train and test — but a model this accurate on one
manipulation method will not generalise as well to methods it never saw
(Face2Face, FaceSwap, NeuralTextures, or in-the-wild videos). Adding more
methods from the same archive is a one-line change to `download_data.py`'s
`WANTED` map if broader coverage is wanted later.

---

## Phase 2 handoff

```python
import sys; sys.path.append("machine-learning")
from inference import load_model, predict

model = load_model("machine-learning/checkpoints/efficientnet_b0_20260901_204509")
prob  = predict(model, face_crop_rgb_ndarray)   # 0.0-1.0 fake probability
```

`predict` accepts a path, a PIL image, an RGB `numpy` array, or a list of those.
`backend/services/detector.py` already targets this API; point
`MODEL_DIR` in `backend/.env` at the chosen checkpoint directory:

```bash
MODEL_DIR=machine-learning/checkpoints/efficientnet_b0_20260901_204509
```

Left at the default (`machine-learning/checkpoints`), the detector falls back to
the most recently written run instead.

Risk thresholds (plan §6: `<40%` real, `40-70%` suspicious, `>70%` high risk)
can be re-tuned from `runs/<run_name>/thresholds.csv`.
