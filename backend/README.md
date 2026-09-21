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
| `NEON_DB_URL` | *(required)* | Neon Postgres connection string (`DATABASE_URL` also accepted). The server has no other database; it will not start without it. |
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
database access is hand-written SQL executed through psycopg2 — there is no
ORM.

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
| `GET` | `/history` | Recent analyses and live sessions, newest first (`?limit=`, default 100, max 500) |
| `POST` | `/live/start` | Open a live screen-share session (201) |
| `POST` | `/live/{session}/frame` | Score one JPEG into a session, get the smoothed verdict |
| `POST` | `/live/{session}/stop` | Finalise a session and store its summary row |
| `GET` | `/live/sessions/{id}` | One stored live session |
| `POST` | `/live/frame` | Score one JPEG, stateless — no session, no smoothing |

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

Uploads and live sessions come from two tables and are unioned into one stream,
so ids are only unique **within** a kind — anything that follows a row has to
carry `kind` with it.

```json
[
  {"id": 3, "kind": "live", "filename": "Live screen share",
   "fake_probability": 0.9911, "status": "HIGH_RISK",
   "created_at": "2026-09-18T06:42:33.565242Z", "has_video": false},
  {"id": 2, "kind": "upload", "filename": "006_002.mp4", "fake_probability": 0.9733,
   "status": "HIGH_RISK", "created_at": "2026-09-01T16:52:36.859098Z",
   "has_video": true}
]
```

### Live screen share

The browser opens a session, posts a 640 px JPEG about twice a second, and
stops. Frames live in memory for the length of a request — nothing is written
to `UPLOAD_DIR` and the bytes never reach the log — and the stored row holds
scores, never images.

```bash
SESSION=$(curl -sX POST http://localhost:8000/live/start | jq -r .session_id)
curl -X POST "http://localhost:8000/live/$SESSION/frame" \
     -H "Content-Type: image/jpeg" --data-binary @frame.jpg
curl -X POST "http://localhost:8000/live/$SESSION/stop"
```

A frame reply carries the raw score, the EMA the badge reads, and the last
60 seconds of smoothed scores for the sparkline:

```json
{"face_found": true, "fake_probability": 0.9922, "smoothed": 0.9911,
 "status": "HIGH_RISK", "frames_scored": 22, "frames_received": 22,
 "dropped": false, "recent": [{"timestamp": 22.4, "fake_probability": 0.9911}]}
```

`dropped: true` means the session was still scoring the previous frame, so this
one was discarded rather than queued and the body repeats the standing verdict.
`/stop` returns the stored `live_sessions` row, or 422 if no face was ever
scored — there is then nothing worth recording. Sessions are process-local and
are reaped after `LIVE_SESSION_TTL` seconds of silence, so an abandoned tab
cannot leak; a multi-instance deployment needs session affinity.

JPEG quality is not a free knob: the classifier reads a known fake as REAL
below about 0.85, so the capture path uses 0.95. See §9 of
[`docs/live-screenshare-refactor.md`](../docs/live-screenshare-refactor.md).

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

The live path reuses the middle of that: `detect_box` + `crop_face` +
`detector.predict` per frame, with MTCNN run only every `LIVE_DETECT_EVERY`
frames and the box reused in between (detection is 2.3× the cost of the rest).
Both paths read the same `RISK_SUSPICIOUS` / `RISK_HIGH` thresholds, so an
upload verdict and a live verdict mean the same thing; only the live badge's
*exit* from a state is sticky, by `RISK_HYSTERESIS`, so a score sitting on a
threshold cannot make it strobe.

The stored tables (`analyses` and `live_sessions`) are the schema in
[`docs/backend-roadmap.md`](../docs/backend-roadmap.md) T4, created by the DDL
in `backend/database.py`, with `frame_scores` stored as `JSONB`.

---

## 4. Smoke test

With the server running and a checkpoint loaded:

```bash
python backend/smoke_test.py                      # defaults to http://127.0.0.1:8000
python backend/smoke_test.py --url http://127.0.0.1:8000 \
    --real data/raw/Real/006.mp4 --fake data/raw/Deepfakes/006_002.mp4
```

It covers both happy paths, `GET /analysis/{id}`, the frame, feedback and rerun
endpoints, a full live session (start, frames, stop, read-back), `/history`
ordering across both kinds, and every error case in the table above. With
the EfficientNet-B0 checkpoint: `006.mp4` → `REAL` 0.0000, `006_002.mp4` →
`HIGH_RISK` 0.9733 with a 0.0–10.2 s suspicious window, 52 frames scored each.

To run against a local Postgres instead of Neon, start one on port 55432 with
a `deepfake` database, then point `backend/.env` at it:

```bash
# backend/.env
NEON_DB_URL=postgresql://postgres:devpass@127.0.0.1:55432/deepfake
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
