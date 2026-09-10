# Backend — Deepfake Detection & Alert System

FastAPI service for **Phase 4** of [`docs/plan.md`](../docs/plan.md) (task-by-task
plan in [`docs/backend-roadmap.md`](../docs/backend-roadmap.md)).

It accepts a video upload, samples frames, crops faces with MTCNN, scores each
crop with the Phase 1 checkpoint (`machine-learning/inference.py`, loaded
in-process — there is no separate ML service), aggregates the frame
probabilities into one risk verdict, stores the analysis in PostgreSQL, and
returns the full result synchronously.

---

## 1. Setup

### 1.1 Dependencies

Reuse the root `.venv` from Phase 1 (it already has torch, torchvision, OpenCV
and facenet-pytorch):

```bash
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt
```

Python 3.10–3.12.

### 1.2 `backend/.env`

Copy `backend/.env.example` to `backend/.env` and fill it in:

| Variable | Default | Meaning |
|---|---|---|
| `NEON_DB_URL` | *(unset)* | Postgres connection string (`DATABASE_URL` also accepted). When unset the server falls back to a local SQLite file at `backend/app.db` so a fresh clone runs without credentials. |
| `MODEL_DIR` | `machine-learning/checkpoints` | Checkpoint directory, relative to the project root. Point it at one run (e.g. `.../efficientnet_b0_20260901_204509`); if it points at the parent, the most recently written run is loaded. |
| `UPLOAD_DIR` | `uploads` | Where uploads are stored, relative to `backend/`. |
| `RISK_SUSPICIOUS` | `0.4` | Below this → `REAL`. |
| `RISK_HIGH` | `0.7` | Above this → `HIGH_RISK`; in between → `SUSPICIOUS`. |
| `FRAME_THRESHOLD` | `0.7` | A frame at/above this counts toward the suspicious region. |
| `MAX_UPLOAD_MB` | `200` | Uploads larger than this are rejected with 413. |
| `CORS_ORIGINS` | `http://localhost:3000,http://127.0.0.1:3000` | Comma-separated origins allowed by CORS (Phase 5 frontend). |

For Neon: create a free project at [neon.tech](https://neon.tech) and copy the
connection string —
`postgresql://USER:PASSWORD@HOST/dbname?sslmode=require`. A
`postgresql+psycopg2://` string also works; the dialect suffix is stripped.

The table is created automatically on startup; there are no migrations. All
database access is hand-written SQL executed through sqlite3 or psycopg2 —
there is no ORM.

### 1.3 Run

```bash
uvicorn backend.main:app --reload --port 8000
```

Swagger UI: <http://localhost:8000/docs>.

On startup the log should show
`Model loaded from .../checkpoints/efficientnet_b0_<ts>`. If the checkpoint is
missing the server still boots in **stub mode** — `predict()` returns a fixed
`0.5` so the frontend can be developed without a model. `GET /health` reports
which mode is active via `model_loaded`.

---

## 2. Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness + whether a real checkpoint is loaded |
| `POST` | `/analyze` | Upload a video, get the full analysis (201) |
| `GET` | `/analysis/{id}` | One stored analysis |
| `GET` | `/analysis/{id}/video` | The source video, for playback in the UI |
| `GET` | `/history` | Recent analyses, newest first (`?limit=`, default 100, max 500) |

### `GET /health`

```json
{"status": "ok", "model_loaded": true}
```

### `POST /analyze`

Multipart form, field name `file`. Allowed extensions: `.mp4`, `.avi`, `.mov`.

```bash
curl -X POST http://localhost:8000/analyze -F "file=@sample_fake.mp4"
```

`201 Created`:

```json
{
  "id": 2,
  "filename": "006_002.mp4",
  "fake_probability": 0.9733,
  "status": "HIGH_RISK",
  "suspicious_start": 0.0,
  "suspicious_end": 10.2,
  "frame_scores": [
    {"timestamp": 0.0, "fake_probability": 0.9891},
    {"timestamp": 0.2, "fake_probability": 0.9812}
  ],
  "created_at": "2026-09-01T16:52:36.859098Z",
  "has_video": true
}
```

- `fake_probability` — mean of the per-frame probabilities (plan §6).
- `status` — `REAL` < `RISK_SUSPICIOUS` ≤ `SUSPICIOUS` ≤ `RISK_HIGH` < `HIGH_RISK`.
- `suspicious_start` / `suspicious_end` — the longest contiguous run of frames at
  or above `FRAME_THRESHOLD`, in seconds; `null` when no run reaches 3 frames
  (plan §7).
- `frame_scores` — one entry per frame that had a usable face, in time order.
- `has_video` — whether the uploaded file is still on disk, i.e. whether
  `GET /analysis/{id}/video` will serve it.

Errors:

| Status | When |
|---|---|
| 400 | Extension not allowed, or the uploaded file is empty |
| 413 | Larger than `MAX_UPLOAD_MB` |
| 422 | Unreadable video, or no face detected in any frame |
| 404 | `GET /analysis/{id}` with an unknown id |

### `GET /analysis/{id}`

Same body as `POST /analyze`. 404 if the id does not exist.

### `GET /analysis/{id}/video`

The video the analysis was run on, served straight from `UPLOAD_DIR` with
`Accept-Ranges: bytes`, so a `<video>` element can seek without downloading the
whole clip. `Content-Disposition` is `inline` under the original filename.

404 when the id is unknown, when the row predates video storage
(`has_video: false`), or when the file has since been removed from disk. `.avi`
is served as `video/x-msvideo`; most browsers will not decode it even though the
pipeline accepts it.

### `GET /history`

```json
[
  {"id": 2, "filename": "006_002.mp4", "fake_probability": 0.9733,
   "status": "HIGH_RISK", "created_at": "2026-09-01T16:52:36.859098Z",
   "has_video": true},
  {"id": 1, "filename": "006.mp4", "fake_probability": 0.0,
   "status": "REAL", "created_at": "2026-09-01T16:52:32.719001Z",
   "has_video": true}
]
```

---

## 3. Pipeline

`backend/services/video_processor.py`, per uploaded video:

1. Sample frames at ~5 fps with OpenCV, uniformly thinned to at most 100 frames
   — the same two-stage sampling `machine-learning/preprocessing.py` used to
   build the training crops.
2. Detect faces per frame with MTCNN; keep the **largest** box above confidence
   0.95 (again matching preprocessing) and skip frames with none.
3. Crop with a 20 px margin. Resize to 224×224 and ImageNet normalization happen
   inside `inference.predict`.
4. Score every crop in **one batched forward pass** via
   `detector.predict_batch`.
5. Aggregate to the mean, derive the status and the suspicious region, persist.

The stored table (`analyses`) is the schema in
[`docs/backend-roadmap.md`](../docs/backend-roadmap.md) T4, created by the DDL
in `backend/database.py`: `frame_scores` is `JSONB` on Postgres, plain `JSON`
on the SQLite fallback.

---

## 4. Smoke test

With the server running and a checkpoint loaded:

```bash
python backend/smoke_test.py                      # defaults to http://127.0.0.1:8000
python backend/smoke_test.py --url http://127.0.0.1:8000 \
    --real data/raw/Real/006.mp4 --fake data/raw/Deepfakes/006_002.mp4
```

It covers both happy paths, `GET /analysis/{id}`, `/history` ordering, and every
error case in the table above. Last run with the EfficientNet-B0 checkpoint:
**22/22 checks passed on Postgres 16 and again on the SQLite fallback** —
`006.mp4` → `REAL` 0.0000, `006_002.mp4` → `HIGH_RISK` 0.9733 with a
0.0–10.2 s suspicious window, 52 frames scored each.

To exercise the Postgres path without a Neon account:

```bash
docker run -d --rm --name df-pg -e POSTGRES_PASSWORD=devpass \
    -e POSTGRES_DB=deepfake -p 55432:5432 postgres:16-alpine
# then in backend/.env:
# NEON_DB_URL=postgresql+psycopg2://postgres:devpass@127.0.0.1:55432/deepfake
```

---

## 5. Phase 5 (frontend) handoff

- Base URL `http://localhost:8000`, Swagger at `/docs`, CORS open for
  `http://localhost:3000`.
- **Upload screen** → `POST /analyze` (multipart, field `file`), synchronous:
  show a spinner; a ~15 s clip takes a few seconds on GPU.
- **Result screen** → `status`, `fake_probability`, `suspicious_start`/`_end`
  for the alert block, `frame_scores` for the per-frame strip.
- **History screen** → `GET /history`; rows carry `has_video` so a play
  affordance can be shown without a second request.
- **Playback** → point a `<video src>` at `GET /analysis/{id}/video` whenever
  `has_video` is true.
