# Backend Roadmap — Deepfake Detection & Alert System

> **Phase 4 of `docs/plan.md`.** Goal: a FastAPI backend that accepts video uploads, runs the Phase 1 model on them, stores every analysis in PostgreSQL, and exposes the 3 endpoints the frontend needs: `POST /analyze`, `GET /analysis/{id}`, `GET /history`.

---

## 1. Goal & Scope

### In scope
- **FastAPI app** with the three Phase 4 endpoints from plan.md §11, plus a `GET /health` check.
- **Neon-hosted PostgreSQL** for analysis history (plan §9).
- **In-process ML inference**: the backend imports the Phase 1 `machine-learning/inference.py` module directly — no separate ML service, one server.
- **Synchronous analysis**: `POST /analyze` runs detection and returns the complete result (frontend just shows a spinner).
- **Risk classification + alert JSON** (plan §6, §8): REAL / SUSPICIOUS / HIGH_RISK.
- **Suspicious timestamps** — the longest contiguous run of frames above the fake-probability threshold (plan §7).

### Out of scope (deferred or deliberately skipped)
- Auth / user accounts, Alembic migrations, Docker, cloud storage, async background jobs, email alerts.
- ML training & evaluation → Phase 1 (`docs/ml-roadmap.md`).
- Frontend → Phase 5.

### Exit criteria (definition of done for Phase 4)
- [ ] Server boots from the root `.venv`, loads the model checkpoint, and serves Swagger at `/docs`.
- [ ] Upload → detect → store in Neon → return result works end-to-end for a real and a fake sample.
- [ ] `GET /analysis/{id}` and `GET /history` return the stored results.
- [ ] Smoke test passes (T9), `backend/README.md` written (T10).

---

## 2. Prerequisites

| Item | Requirement | Notes |
|---|---|---|
| Python | **3.10 – 3.12** | Same as Phase 1 |
| Phase 1 checkpoint | `machine-learning/checkpoints/<model>/best.pth` + `config.json` | Created by `docs/ml-roadmap.md`; needed for T5 |
| Neon account | **Free tier** | neon.tech — gives you a Postgres connection string |
| `curl` | Any | For the T9 smoke test |

---

## 3. Environment Setup

### 3.1 Project structure (created as you go)

```text
dbms-project/
├── backend/                  # Phase 4 — this roadmap
│   ├── main.py               # FastAPI app, CORS, startup: create tables + load model
│   ├── config.py             # reads backend/.env (pydantic-settings)
│   ├── database.py           # engine, SessionLocal, Base, get_db
│   ├── models.py             # SQLAlchemy Analysis model
│   ├── schemas.py            # Pydantic request/response models
│   ├── routes/
│   │   ├── analyze.py        # POST /analyze
│   │   └── history.py        # GET /analysis/{id}, GET /history
│   ├── services/
│   │   ├── detector.py       # wraps machine-learning/inference.py (load_model, predict)
│   │   └── video_processor.py# frames → crops → per-frame probs → aggregate
│   ├── uploads/              # gitignored — uploaded videos
│   ├── requirements.txt
│   └── README.md             # T10
│
├── machine-learning/         # Phase 1 — consumed via inference.py
│   ├── checkpoints/          # gitignored
│   └── inference.py          # load_model(name) / predict(model, crop)
│
├── frontend/                 # Phase 5
├── docs/
│   ├── plan.md
│   ├── ml-roadmap.md
│   └── backend-roadmap.md    # this file
└── .gitignore
```

### 3.2 Virtual environment

Reuse the root `.venv` from Phase 1 (it already has torch, torchvision, opencv, MTCNN — the backend calls into those via `machine-learning/inference.py`).

```bash
cd "C:\Users\Mudit Garg\Desktop\dbms-project"
.venv\Scripts\activate          # Windows
pip install -r backend/requirements.txt
```

### 3.3 `backend/requirements.txt`

```text
fastapi>=0.110
uvicorn[standard]>=0.29
python-multipart>=0.0.9
SQLAlchemy>=2.0
psycopg2-binary>=2.9
pydantic-settings>=2.1
python-dotenv>=1.0
```

No Alembic, no docker, no async DB driver — keep it simple.

### 3.4 Neon database

1. Create a free project at [neon.tech](https://neon.tech).
2. Copy the connection string from the dashboard.
3. Create `backend/.env`:

```bash
DATABASE_URL=postgresql+psycopg2://USER:PASSWORD@HOST/db?sslmode=require
MODEL_DIR=../machine-learning/checkpoints/efficientnet_b0_<ts>/
UPLOAD_DIR=uploads
RISK_SUSPICIOUS=0.4
RISK_HIGH=0.7
FRAME_THRESHOLD=0.7
MAX_UPLOAD_MB=200
```

Add to `.gitignore`:

```gitignore
backend/.env
backend/uploads/
```

---

## 4. Task Checklist

Progress tracker — check boxes off as each task completes.

- [x] **T0 — Repo hygiene**: `.env` + `uploads/` ignored
- [x] **T1 — Environment**: deps installed, uvicorn boots a hello app
- [x] **T2 — Neon DB**: connection from Python works (`SELECT 1`)
- [x] **T3 — Config + DB plumbing**: tables auto-created at startup
- [x] **T4 — Schema**: `Analysis` model + Pydantic schemas
- [x] **T5 — Detector service**: checkpoint loads at startup, `predict()` returns 0–1
- [ ] **T6 — Video processing**: per-frame probs → mean score → risk status → suspicious region
- [ ] **T7 — Routes**: all 3 endpoints wired in `main.py` with CORS
- [ ] **T8 — Validation & errors**: extension/size/empty-video cases
- [ ] **T9 — Smoke test**: real + fake samples through every endpoint
- [ ] **T10 — Docs**: `backend/README.md` written, checklist finalized

---

### T0 — Repo hygiene

Add to `.gitignore`:

```gitignore
backend/.env
backend/uploads/
backend/__pycache__/
```

**Done when**: `git status` stays clean while `backend/uploads/` grows.

---

### T1 — Environment

- Install `backend/requirements.txt` (§3.3).
- Boot a minimal app to confirm uvicorn works:

```bash
uvicorn backend.main:app --reload --port 8000
```

**Done when**: server starts and responds at `http://localhost:8000/health` (a stub `{"status": "ok"}` is fine at this stage).

---

### T2 — Neon DB

- Create the project and paste the connection string into `backend/.env` (§3.4).
- Verify the connection from Python:

```bash
python -c "
import os
from dotenv import load_dotenv
import psycopg2
load_dotenv('backend/.env')
conn = psycopg2.connect(os.environ['DATABASE_URL'])
print(conn.cursor().execute('SELECT 1').fetchall())
"
```

**Done when**: `[(1,)]` prints — Neon is reachable from your machine.

---

### T3 — Config + DB plumbing

- `config.py` — pydantic-settings `Settings` class reading `backend/.env` (`DATABASE_URL`, `MODEL_DIR`, `UPLOAD_DIR`, thresholds, `MAX_UPLOAD_MB`).
- `database.py` — `engine = create_engine(Settings().DATABASE_URL)`, `SessionLocal`, `Base`, and a `get_db()` dependency that yields a session.
- In `main.py`, use the FastAPI lifespan to call `Base.metadata.create_all(engine)` on startup.

**Done when**: server boots and the `analyses` table appears in the Neon console (empty, but present).

---

### T4 — Schema

SQLAlchemy `Analysis` model (mirrors the plain SQL below):

```sql
CREATE TABLE analyses (
    id SERIAL PRIMARY KEY,
    filename TEXT NOT NULL,
    fake_probability REAL NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('REAL','SUSPICIOUS','HIGH_RISK')),
    suspicious_start REAL,
    suspicious_end REAL,
    frame_scores JSONB NOT NULL DEFAULT '[]',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

`schemas.py` Pydantic models:
- `AnalysisOut` — the full result returned to the frontend: `id, filename, fake_probability, status, suspicious_start, suspicious_end, frame_scores, created_at`.
- `HistoryOut` — `id, filename, fake_probability, status, created_at` (the history list row, plan §5).

**Done when**: inserting a row via `SessionLocal` works and it shows up in Neon.

---

### T5 — Detector service (`backend/services/detector.py`)

- At startup (lifespan), add `machine-learning/` to `sys.path` and call `inference.load_model(Settings().MODEL_DIR)`.
- Expose `predict(crop) → float` (0–1 fake probability) as a module-level wrapper around `inference.predict()`.

```python
# detector.py (sketch)
import sys, os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2] / "machine-learning"))
import inference

MODEL = None

def load_model(model_dir: str):
    global MODEL
    MODEL = inference.load_model(model_dir)   # reads config.json, loads best.pth

def predict(crop) -> float:
    return inference.predict(MODEL, crop)     # → 0.0–1.0
```

- **Stub fallback**: if the checkpoint dir is missing, log a warning and set `MODEL = None`; `predict()` then returns a fixed `0.5` so the frontend and routes can still be developed. This is a documented dev fallback, not a silent production behavior.

**Done when**: server startup logs "Model loaded from <path>" with the real checkpoint, and `predict()` on one test crop returns a value in [0, 1].

---

### T6 — Video processing (`backend/services/video_processor.py`)

Per uploaded video (mirrors Phase 2/3 of plan.md):

1. **Extract frames** at ~5 fps with `cv2.VideoCapture`, cap at ~100 frames (uniform sample if longer).
2. **Detect faces** per frame with the same MTCNN approach used in Phase 1 preprocessing; skip frames with no face.
3. **Crop + resize** each face to 224×224 (ImageNet normalization, matching training).
4. **Score** each crop with `detector.predict()` → list of `{timestamp, fake_probability}`.
5. **Aggregate**: video `fake_probability = mean(frame probs)` (plan §6).
6. **Risk status** (plan §6): `< 0.4` → `REAL`, `0.4–0.7` → `SUSPICIOUS`, `> 0.7` → `HIGH_RISK`.
7. **Suspicious region** (plan §7): the longest contiguous run of frames with `fake_probability ≥ FRAME_THRESHOLD` → `(suspicious_start, suspicious_end)` in seconds; `None` if no run of ≥ 3 frames.

Return:

```json
{
  "filename": "video_001.mp4",
  "fake_probability": 0.91,
  "status": "HIGH_RISK",
  "suspicious_start": 8.0,
  "suspicious_end": 12.0,
  "frame_scores": [
    {"timestamp": 0.0, "fake_probability": 0.12},
    {"timestamp": 8.0, "fake_probability": 0.88},
    {"timestamp": 12.0, "fake_probability": 0.85}
  ]
}
```

**Done when**: running the processor on one short sample video (with a real checkpoint) returns sane scores — high for a fake sample, low for a real one.

---

### T7 — Routes

- `main.py`: FastAPI app, `CORSMiddleware` allowing `http://localhost:3000` (Phase 5 frontend), lifespan starts DB + model.
- `routes/analyze.py` — `POST /analyze` (multipart `UploadFile`):
  1. Save file to `backend/uploads/` (uuid-prefixed name).
  2. Run `video_processor` on it.
  3. Insert an `Analysis` row.
  4. Return the full result (201).
- `routes/history.py`:
  - `GET /analysis/{id}` — one row, 404 if missing.
  - `GET /history` — all rows, `ORDER BY created_at DESC`, limit ~100 (plan §5 history table).
- `GET /health` — `{"status": "ok"}`.

**Done when**: all 3 endpoints + `/health` respond correctly from Swagger at `/docs`.

---

### T8 — Validation & errors

- Allowed extensions: `.mp4`, `.avi`, `.mov` (plan §2) → anything else returns **400**.
- Size limit from `MAX_UPLOAD_MB` → larger returns **413**.
- Empty video / no faces detected → **422** with a clear message.
- Unknown `id` → **404**.

**Done when**: each case above returns the expected status code, not a 500.

---

### T9 — Smoke test

```bash
curl -X POST http://localhost:8000/analyze -F "file=@sample_fake.mp4"
curl -X POST http://localhost:8000/analyze -F "file=@sample_real.mp4"
curl http://localhost:8000/analysis/1
curl http://localhost:8000/history
```

**Done when**:
- fake sample → `HIGH_RISK`, high probability, non-empty suspicious region;
- real sample → `REAL`, low probability;
- `/analysis/1` matches the upload response;
- `/history` shows both rows, newest first;
- the rows exist in the Neon console (`SELECT * FROM analyses;`).

---

### T10 — Docs (`backend/README.md`)

- Setup: venv, `.env` variables, Neon instructions (brief).
- Run: `uvicorn backend.main:app --reload --port 8000`.
- Endpoints table with request/response shapes.
- Link to `docs/plan.md` and this roadmap.

**Done when**: a fresh clone + these steps reproduces the running server.

---

## 5. Verification Strategy

| Stage | Check |
|---|---|
| Boot | Server starts, model loads, tables exist in Neon |
| Happy path | Real sample → REAL/low prob; fake sample → HIGH_RISK/high prob; by-id and history match |
| DB persistence | `SELECT * FROM analyses;` in Neon console shows rows with valid JSONB |
| Edge cases | Bad extension → 400, oversized → 413, no-face video → 422, unknown id → 404 |
| Stub mode | Server runs without a checkpoint, `predict()` returns 0.5 |

If a fake sample comes back REAL: check the checkpoint path in `.env`, face-crop quality, and whether frame extraction is actually sampling the video.

---

## 6. Phase 5 Handoff

Phase 5 (frontend) receives:

- **Server**: `http://localhost:8000`, Swagger docs at `/docs`, CORS already open for `localhost:3000`.
- **Upload screen**: `POST /analyze` (multipart) → full result JSON; synchronous, so just show a spinner while waiting.
- **Result screen**: `status`, `fake_probability`, `suspicious_start`/`suspicious_end` (alert block from plan §8), and `frame_scores` for the frame strip.
- **History screen**: `GET /history` → list of `{id, filename, fake_probability, status, created_at}`.

---

## 7. Reference — plan.md crosswalk

| plan.md section | Covered by |
|---|---|
| §6 Risk classification (40/70 thresholds) | T6 |
| §7 Suspicious timestamps | T6 |
| §8 Alert JSON | T6, T7 |
| §9 PostgreSQL + local storage | T2, T3, T7 |
| §10 Folder structure | §3.1 |
| §11 Phase 4 (the 3 endpoints) | T3–T7 |
