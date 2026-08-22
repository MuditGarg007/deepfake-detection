# ML Training Roadmap — Deepfake Detection & Alert System

> **Phase 1 of `docs/plan.md`.** Goal: a trained face-crop classifier that maps `image → real/fake probability`, delivered as a reusable model checkpoint + inference API for Phase 2 (video processing).

---

## 1. Goal & Scope

### In scope
- **Two fine-tuned CNN models**: `efficientnet_b0` and `xception` (plan §3, §13).
- **Binary classification**: real (original) vs fake (DeepFakes c23).
- **Face-crop pipeline**: video → frames → MTCNN face crop → 224×224 → dataset.
- **Evaluation**: accuracy, precision, recall, F1, ROC-AUC, confusion matrix, inference time (plan §12).
- **Model comparison** report (plan §13).

### Out of scope (deferred to later phases)
- Video-level prediction & suspicious timestamps → Phase 2/3.
- Backend API, alerting, frontend → Phases 4/5.
- Audio, multimodal, temporal transformers, adversarial training (plan final scope).

### Exit criteria (definition of done for Phase 1)
- [ ] Checkpoint exists for **both** models, each with a `config.json` and metrics report.
- [ ] Test-set ROC-AUC **≥ 0.90** on FF++ c23 (target; see §7 verification).
- [ ] `inference.py` exposes `load_model(name)` + `predict(crop) → probability` and passes a smoke test.
- [ ] `machine-learning/runs/comparison.md` compares both models (Accuracy / F1 / AUC table).

---

## 2. Prerequisites

| Item | Requirement | Notes |
|---|---|---|
| Python | **3.10 – 3.12** | Match your torch version's wheel support |
| GPU | **NVIDIA GPU with CUDA** | Check driver with `nvidia-smi` |
| CUDA toolkit | Optional (bundled with torch wheels) | Install torch via `pip` CUDA build |
| Disk space | **~25–35 GB free** | FF++ c23 download + extracted videos + processed crops |
| RAM | 16 GB+ recommended | Face detection is the memory-heavy step |

---

## 3. Environment Setup

### 3.1 Project structure (created as you go)

```text
dbms-project/
├── data/                      # gitignored — raw + processed data
│   ├── raw/
│   │   ├── original_sequences/       # FF++ real videos
│   │   └── manipulated_sequences/    # FF++ DeepFakes videos
│   └── processed/                    # face-crop JPEGs
│       ├── train/{real,fake}/
│       ├── val/{real,fake}/
│       └── test/{real,fake}/
│
├── backend/                   # FastAPI app (Phase 4)
│   └── ...
│
├── frontend/                  # Next.js app (Phase 5)
│   └── ...
│
├── machine-learning/          # everything ML lives here (Phase 1)
│   ├── requirements.txt
│   ├── preprocessing.py       # T3: frame extraction + face crops
│   ├── dataset.py             # T4: Dataset + augmentations
│   ├── models.py              # T5: EfficientNet-B0 / Xception builders
│   ├── train.py               # T6: training loop + early stopping
│   ├── evaluate.py            # T7: test metrics + plots
│   ├── inference.py           # T8: load_model + predict
│   ├── compare.py             # T9: model comparison
│   ├── checkpoints/           # gitignored — .pth + config.json per run
│   ├── runs/                  # gitignored — logs, plots, metrics.json
│   └── README.md              # T10: reproduction steps
│
├── docs/
│   ├── plan.md
│   └── ml-roadmap.md          # this file
└── .gitignore                 # T0
```

### 3.2 Virtual environment

```bash
cd "C:\Users\Mudit Garg\Desktop\dbms-project"
python -m venv .venv
.venv\Scripts\activate          # Windows
```

### 3.3 `machine-learning/requirements.txt`

```text
torch>=2.1
torchvision>=0.16
opencv-python>=4.8
facenet-pytorch>=2.5            # MTCNN face detection
timm>=0.9                        # efficientnet_b0, xception
scikit-learn>=1.3
matplotlib>=3.8
seaborn>=0.13
pandas>=2.0
numpy>=1.24
Pillow>=10.0
tqdm>=4.66
```

Install:

```bash
pip install -r machine-learning/requirements.txt
```

### 3.4 GPU sanity check

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expect: `2.x True <GPU name>`. If `False`, stop here — fix the CUDA torch install before proceeding.

---

## 4. Task Checklist

Progress tracker — check boxes off as each task completes.

- [ ] **T0 — Repo hygiene**: create `.gitignore`
- [ ] **T1 — Environment**: deps installed, GPU + AMP verified
- [ ] **T2 — Dataset**: FF++ downloaded **or** fallback decision recorded
- [ ] **T3 — Preprocessing**: `data/processed/` + `manifest.csv` built
- [ ] **T4 — Dataset loader**: augmentations working
- [ ] **T5 — Model builders**: both models instantiate, forward pass OK
- [ ] **T6 — Train**: EfficientNet-B0 trained → checkpoint + config
- [ ] **T7 — Evaluate**: EfficientNet-B0 metrics + plots
- [ ] **T6b — Train**: Xception trained → checkpoint + config
- [ ] **T7b — Evaluate**: Xception metrics + plots
- [ ] **T8 — Inference**: `predict()` smoke test passes
- [ ] **T9 — Compare**: `machine-learning/runs/comparison.md` written
- [ ] **T10 — Docs**: `machine-learning/README.md` written, this checklist finalized

---

### T0 — Repo hygiene

Create `.gitignore` at repo root:

```gitignore
.venv/
__pycache__/
*.pyc
data/
machine-learning/checkpoints/
machine-learning/runs/
*.log
```

**Done when**: `git status` stays clean while `data/` and `machine-learning/runs/` grow.

---

### T1 — Environment

- Install `machine-learning/requirements.txt` (§3.3).
- Verify GPU (§3.4).
- AMP smoke test (T6 trains with AMP; do this early):

```bash
python -c "
import torch
m = torch.nn.Linear(64, 2).cuda()
x = torch.randn(8, 64).cuda()
with torch.autocast('cuda', dtype=torch.float16):
    y = m(x)
print('AMP OK', y.shape, y.dtype)
"
```

**Done when**: GPU detected, AMP forward pass runs, no CUDA errors.

---

### T2 — Dataset (FaceForensics++)

**Primary path (recommended):**

1. **Create an account / request access** on the [FaceForensics++ download page](https://github.com/ondyari/FaceForensics) (access requires a form approval).
2. Download **c23** quality (not raw):
   - `original_sequences/youtube` → **real** videos
   - `manipulated_sequences/deepfakes` → **fake** videos
   - Use the official `download_samples.py` / `download_dataset.py` scripts or a per-video downloader for a subset.
3. **Suggested subset for v1**: 500 real + 500 fake videos. Scaling up to 1000+1000 is possible later — the pipeline is identical.
4. Organize into:

```text
data/raw/original_sequences/youtube/c23/videos/*.mp4
data/raw/manipulated_sequences/deepfakes/c23/videos/*.mp4
```

> **Decision gate ⚠️**: If the FF++ access request is slow/denied or you hit storage limits, **switch to a Kaggle pre-cropped faces dataset** (e.g. "140k Real and Fake Faces"). The preprocessing step T3 is then *skipped or reduced to resize-only* — crops are already 224×224. **Record the choice** in `machine-learning/README.md` under "Dataset used".

**Done when**: videos organized under `data/raw/`, real and fake clearly separated, count logged.

---

### T3 — Preprocessing (`machine-learning/preprocessing.py`)

Creates the face-crop dataset once, on disk. Training later reads JPEGs directly (no per-epoch face detection).

Pipeline per video (matches plan §5):

1. **Split first at the video level** — 70/15/15 (train/val/test) by video path (NOT by frame). Prevents frame leakage between splits. Record in `manifest.csv`.
2. **Extract frames** at **5 fps** (plan §5), cap at **50 frames/video** (uniform sampling if longer). Use `cv2.VideoCapture`.
3. **Detect faces** with **MTCNN** (`facenet-pytorch`):
   - Confidence **≥ 0.95**; if multiple faces, take the largest.
   - No face detected → skip frame (log skip rate).
4. **Crop + resize** to **224×224** with a margin of ~20 px around the box.
5. **Save** as JPEG to `data/processed/{split}/{label}/{video}_{frame_idx}.jpg`.
6. Append to `data/processed/manifest.csv`:

```csv
video,split,label,frame_idx,path
video_001.mp4,train,real,12,data/processed/train/real/video_001_12.jpg
```

CLI sketch:

```bash
python machine-learning/preprocessing.py --raw data/raw --out data/processed \
    --fps 5 --max-frames 50 --margin 20 --conf 0.95 --seed 42
```

**Done when**: `data/processed/` has all 6 split/label subdirs populated, `manifest.csv` valid, and a quick visual spot-check of ~10 crops shows correctly cropped faces.

---

### T4 — Dataset loader (`machine-learning/dataset.py`)

- `FaceDataset(Dataset)` reads crops from `manifest.csv` (or from folder structure).
- **Augmentation (train only)**:
  - `RandomHorizontalFlip` (p=0.5)
  - `RandomAffine` (rotate ±10°, translate ±5%)
  - `ColorJitter` (brightness 0.2, contrast 0.2)
- **Normalization**: ImageNet mean/std `(0.485, 0.456, 0.406)` / `(0.229, 0.224, 0.225)` — required for pretrained weights.
- DataLoaders: `batch_size 64` (GPU), `num_workers 4`, `pin_memory True`, `drop_last` on train.

**Done when**: a single training step consumes one batch without error; augmentation visibly changes images.

---

### T5 — Model builders (`machine-learning/models.py`)

```python
import timm, torch.nn as nn

def get_model(name: str, num_classes: int = 1) -> nn.Module:
    model = timm.create_model(name, pretrained=True, num_classes=0)  # drop head
    model = nn.Sequential(
        model,                       # backbone → features
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(model.num_features, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(128, 1),           # single logit
    )
    return model
```

- Supported names: `"efficientnet_b0"`, `"xception"` (both available in timm).
- Single logit → `BCEWithLogitsLoss` (plan §5).
- Weight init: leave pretrained backbone as-is; the new head starts random (default PyTorch init).

**Done when**: `get_model("efficientnet_b0")` and `get_model("xception")` both instantiate and pass a dummy forward + backward.

---

### T6 — Training (`machine-learning/train.py`)

```bash
python machine-learning/train.py --model efficientnet_b0 --data data/processed \
    --epochs 15 --batch-size 64 --lr 1e-4 --device cuda --amp \
    --patience 5 --seed 42
```

Training loop:
- **Optimizer**: Adam, `lr=1e-4`, weight decay `1e-5`.
- **Loss**: `BCEWithLogitsLoss`.
- **AMP**: `torch.autocast('cuda')` + `GradScaler`.
- **Gradient clipping**: `clip_grad_norm_(1.0)`.
- **Full fine-tune** (all backbone weights trainable).
- **Early stopping** on **val ROC-AUC**, patience 5.
- **Checkpointing**: save best model to

```text
machine-learning/checkpoints/<model>_<timestamp>/best.pth
machine-learning/checkpoints/<model>_<timestamp>/config.json
```

`config.json` captures: model name, args, data split counts, class balance, augmentation flags, best val metrics.

- **Logging**: per-epoch CSV to `machine-learning/runs/<model>_<timestamp>/log.csv` (train loss/acc, val loss/acc/AUC).

**Sequence**: train EfficientNet-B0 fully (T6+T7), then Xception (T6b+T7b).

**Done when**: training finishes with early stop, `best.pth` + `config.json` saved, loss curves look sane (val AUC rising, no divergence).

---

### T7 — Evaluation (`machine-learning/evaluate.py`)

```bash
python machine-learning/evaluate.py --checkpoint machine-learning/checkpoints/efficientnet_b0_*/ --data data/processed
```

Outputs to `machine-learning/runs/<model>_<timestamp>/`:
- `metrics.json`: **accuracy, precision, recall, F1, ROC-AUC** (+ inference time per image).
- `roc_curve.png`
- `confusion_matrix.png`
- `thresholds.csv`: precision/recall per threshold (for Phase 2's 40/70 risk cutoffs, plan §6).

**Done when**: metrics file + both plots exist; numbers are recorded in `machine-learning/README.md`.

---

### T8 — Inference (`machine-learning/inference.py`)

```python
from inference import load_model, predict

model = load_model("machine-learning/checkpoints/efficientnet_b0_<ts>/")   # reads config.json
prob = predict(model, "data/processed/test/fake/xxx.jpg")     # → 0.0–1.0 fake probability
```

- `load_model(name)` → builds arch from config, loads `best.pth`, sets eval mode, moves to `cuda` if available.
- `predict(model, image_path_or_ndarray)` → applies preprocessing (resize 224, normalize) → returns `sigmoid(logit)`.
- Smoke test: run over 100 test crops; expect no errors, probabilities within [0,1], ~0.9 on fake samples.

**Done when**: smoke test passes; this module is the interface Phase 2 consumes.

---

### T9 — Comparison (`machine-learning/compare.py`)

```bash
python machine-learning/compare.py --checkpoints machine-learning/checkpoints/efficientnet_b0_*/ machine-learning/checkpoints/xception_*/
```

- Builds the table from each model's `metrics.json` (plan §13):

| Model | Accuracy | F1 | ROC-AUC | Infer ms/img |
|---|---|---|---|---|
| EfficientNet-B0 | … | … | … | … |
| Xception | … | … | … | … |

- Writes `machine-learning/runs/comparison.md` with the table + a one-paragraph conclusion (which model wins, by which metric).

**Done when**: `comparison.md` exists and is cited in the project report.

---

### T10 — Docs (`machine-learning/README.md`)

- Reproduction steps: env setup → dataset → preprocessing → train → evaluate → infer.
- Record **which dataset was used** (FF++ or Kaggle fallback) and its size.
- Record hardware used + total training time.
- Link to `docs/plan.md` and this roadmap.

**Done when**: a fresh clone + `machine-learning/README.md` steps reproduce a training run.

---

## 5. Verification Strategy

| Stage | Check |
|---|---|
| Smoke test | 1–2 epochs on ~1k images (subset) — pipeline runs end-to-end |
| Full run | Val ROC-AUC ≥ **0.90** (FF++ c23 target; typical reported 0.92–0.97) |
| Overfitting sanity | Train vs val accuracy gap < ~15 pts; early stopping triggered |
| Inference latency | < 50 ms/img on GPU (report in comparison) |

If val AUC < 0.85: check class balance, face-crop quality, frame sampling; increase epochs (with patience), or add more videos.

---

## 6. Phase 2 Handoff

Phase 2 (video processing) receives:

- **Checkpoint path**: `machine-learning/checkpoints/<best_model>/best.pth` + `config.json`
- **API**: `inference.load_model()` / `inference.predict()` → per-frame fake probability
- **Aggregation**: mean of frame probabilities → video score; risk thresholds from plan §6 (`< 40%` Real, `40–70%` Suspicious, `> 70%` High Risk) tuned on the test set via `thresholds.csv`
- **Suspicious timestamps** (Phase 3) reuse the per-frame probabilities directly.

---

## 7. Reference — plan.md crosswalk

| plan.md section | Covered by |
|---|---|
| §3 Model (EfficientNet/Xception) | T5, T6 |
| §4 Dataset (FF++) | T2, T3 |
| §5 Training process (frames→faces→224→CNN→sigmoid) | T3, T5, T6 |
| §11 Phase 1 (dataset→face extraction→training→evaluation) | T2–T7 |
| §12 Metrics (acc/precision/recall/F1/AUC/inference) | T7 |
| §13 Model comparison | T9 |
