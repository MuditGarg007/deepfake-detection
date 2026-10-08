# Live Screen-Share Detection — Refactor Plan

> **Phase 6.** Goal: keep the existing upload flow working, and add a second way in — the viewer shares a screen or a tab, and the app scores whatever face is on it, live, while it plays.

---

## 1. Goal & Scope

### In scope
- **Screen capture in the browser** via `getDisplayMedia`, driven by a custom Streamlit component.
- **A per-frame scoring endpoint** on the existing FastAPI service, reusing the model and face pipeline that `POST /analyze` already uses.
- **Live verdicts**: a smoothed fake probability and a REAL / SUSPICIOUS / HIGH_RISK badge that updates a couple of times per second.
- **A session summary** written to Postgres when the share stops, so live runs appear in history next to uploaded files.
- **One scoring definition** shared by both paths, so the upload verdict and the live verdict can never drift apart.

### Out of scope
- Audio, multi-face tracking, or per-speaker attribution.
- Recording the shared screen. Frames are scored and dropped.
- Replacing the upload flow. `POST /analyze` keeps its current contract.
- In-browser inference (see §12).

### Exit criteria
- [x] `POST /analyze` behaviour is byte-for-byte unchanged after the refactor — smoke test still passes. The known fake still scores 0.9733 HIGH_RISK, the known real 0.0 REAL.
- [x] Starting a share in the Streamlit app produces a live badge that tracks the shared content.
- [x] Stopping the share writes one `live_sessions` row and shows it in history.
- [x] Sustained throughput of at least 1 verdict/second — see the T4 measurement in §9.

---

## 2. Where the code is today

| Piece | File | What it does |
|---|---|---|
| Upload endpoint | `backend/routes/analyze.py:59` | Validates extension, streams the file to `UPLOAD_DIR`, calls `process_video`, inserts one row |
| Whole-video pipeline | `backend/services/video_processor.py:141` | `process_video`: extract → MTCNN → crop → batch predict → aggregate |
| Model wrapper | `backend/services/detector.py` | Loads the checkpoint once, `predict` / `predict_batch`, falls back to a 0.5 stub |
| Frontend | `streamlit-frontend/app.py` | File uploader, one POST, result panel and chart |
| Schema | `backend/database.py:6` | A single `analyses` table |

The shape of the problem: `process_video` is one function that opens a file, samples frames, detects faces, scores them, and aggregates a verdict. Live capture needs the middle of that — score one frame that arrived over the wire — without the file I/O at either end.

---

## 3. Target architecture

```
browser                          FastAPI                        Postgres
───────                          ───────                        ────────
getDisplayMedia
   │
   ├─ <video> → <canvas> @ 2 fps
   │     └─ toBlob(jpeg, 0.7)
   │            │
   │            ▼
   │      POST /live/{session}/frame ──► score_frame(rgb)
   │                                        │  MTCNN (every 5th frame)
   │                                        │  crop + detector.predict
   │                                        ▼
   │            ◄── {probability, smoothed, status}
   │                                     LiveSession (in memory)
   │                                        ring buffer + EMA
   ▼
 badge + sparkline            POST /live/{session}/stop ──────► live_sessions
```

Six pieces, each replaceable on its own:

1. **Capture** — browser-side, decides *when* a frame is worth sending.
2. **Transport** — carries JPEG bytes up and a verdict down.
3. **Scoring** — `score_frame`, shared with the upload path.
4. **Session state** — in-memory smoothing and history for one share.
5. **Persistence** — one row per finished session.
6. **UI** — badge, sparkline, start/stop.

---

## 4. Backend refactor — `score_frame`

The single structural change. `video_processor.process_video` splits into three functions that compose back into the old behaviour:

```python
def score_frame(rgb, face_detector=None) -> FrameScore | None:
    """Detect the largest face in one RGB frame and score it.
    Returns None when no face clears FACE_CONF."""

def aggregate(scores: list[FrameScore]) -> tuple[float, str, float | None, float | None]:
    """Mean probability, risk status, suspicious region. Lifted verbatim
    from the tail of process_video."""

def process_video(path, filename=None) -> VideoResult:
    """Unchanged signature and return value — now extract_frames +
    score_frame per frame + aggregate."""
```

Notes on doing this safely:

- **Keep batching on the upload path.** `detector.predict_batch` is meaningfully faster than N single predictions, so `process_video` should still collect crops and batch them. That means `score_frame` needs a sibling — `detect_face(rgb) -> crop | None` — and `process_video` calls `detect_face` in a loop, then batches. `score_frame` is `detect_face` + `predict`, for callers with exactly one frame.
- **MTCNN stays a module-level singleton** (`get_mtcnn`). Two paths, one detector instance.
- `FACE_MARGIN`, `FACE_CONF`, `MIN_RUN_FRAMES` and the risk thresholds stay where they are. Live must not invent its own constants.
- The existing smoke test (`backend/smoke_test.py`) is the regression gate for this step. It should pass before and after with no edits.

---

## 5. Capture — a custom Streamlit component

Streamlit has no access to `getDisplayMedia`, and its rerun model — the whole script re-executes on every interaction — is the wrong shape for a 2 fps stream. So capture lives in a small JS component under `streamlit-frontend/components/screenshare/`, and it talks to the backend directly rather than round-tripping through Python.

The loop is short:

```js
const stream = await navigator.mediaDevices.getDisplayMedia({ video: true });
video.srcObject = stream;
setInterval(async () => {
  if (inFlight) return;               // drop-latest: never queue
  ctx.drawImage(video, 0, 0, W, H);   // W scaled to 640px wide
  const blob = await new Promise(r => canvas.toBlob(r, "image/jpeg", 0.95));
  inFlight = true;
  const verdict = await post(blob);
  inFlight = false;
  render(verdict);
}, 500);
```

Python's side of the component handles only session start/stop and receives the final summary via `Streamlit.setComponentValue`.

**Measured during T3 — the iframe can capture, but only because it is
same-origin.** Streamlit's component `allow` attribute lists camera and
microphone and *not* `display-capture`, which looks fatal. It is not: a
component declared with `path=` is served by the Streamlit server itself, so
the frame is same-origin with the page, and `display-capture`'s default
allowlist of `self` covers it. `document.featurePolicy.allowsFeature(
"display-capture")` reads true inside the frame and `getDisplayMedia` works.
A component hosted on its own origin — a dev server, a CDN — would be denied,
and Streamlit gives us no way to add the token.

Two smaller things that cost time:

- **The component posts from the browser**, so the backend's `CORS_ORIGINS`
  has to contain the Streamlit origin. `backend/.env` overrides the default in
  `config.py`, so it needs the entry too, not just the code default.
- **Frame height must be measured from `document.body`.** Inside a component
  iframe `documentElement.scrollHeight` is the iframe's own height, which only
  ratchets upward, so the panel never shrinks back after the preview is hidden. A `ResizeObserver` on the body keeps it honest.

**Why not `streamlit-webrtc`.** It is built around `getUserMedia` and a webcam source; pointing it at a display surface means fighting the library, and it brings a WebRTC signalling path we do not otherwise need. The hand-rolled loop above is roughly 80 lines and leaves the sampling rate, resolution and backpressure policy in our hands — which §9 shows are the only knobs that matter here.

---

## 6. Transport

**v1 — HTTP.** `POST /live/{session_id}/frame` with the JPEG as the body, replying with the verdict JSON. One request per sampled frame, ~2/second. No new infrastructure, works through Cloud Run exactly as deployed today.

**v2 — WebSocket.** `WS /live/{session_id}` if per-request overhead shows up in the latency numbers. Cloud Run supports WebSockets with a 60-minute connection cap, which is longer than any plausible session. Defer until v1 is measured; do not build both.

**Backpressure is a requirement, not a tuning detail.** If a verdict has not come back, the next frame is dropped, not queued — on the client via the `inFlight` flag, and on the server via a single-slot worker that replaces any waiting frame. A queue that grows means the badge starts describing the past, which is worse than a badge that updates less often.

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/live/start` | Mints a `session_id`, creates the `LiveSession` |
| `POST` | `/live/{id}/frame` | Scores one JPEG, returns the current verdict |
| `POST` | `/live/{id}/stop` | Finalises, writes the row, returns the summary |

There is also `POST /live/frame` — the stateless one-frame endpoint from T2,
kept for one-shot checks against a still. The UI uses the session endpoints.

Response body for a frame:

```json
{
  "fake_probability": 0.71,
  "smoothed": 0.63,
  "status": "SUSPICIOUS",
  "face_found": true,
  "frames_scored": 48,
  "frames_received": 51,
  "dropped": false,
  "recent": [{"timestamp": 23.5, "fake_probability": 0.61}]
}
```

`recent` is the ring buffer of §7 — the last 60 seconds of smoothed scores,
which is what the client's sparkline draws, so the client holds no history of
its own. `dropped` marks a frame the server refused to score because the
single slot was busy; the body then carries the verdict that already stands
and the client leaves the badge alone.

---

## 7. Session state, smoothing and hysteresis

`LiveSession`, held in a process-local dict keyed by session id:

| Field | Purpose |
|---|---|
| `scores` | Ring buffer, last ~120 entries (60 s at 2 fps) |
| `ema` | Exponential moving average, alpha ≈ 0.2 |
| `status` | Current badge state, updated with hysteresis |
| `last_box` | Reused between MTCNN runs (§9) |
| `started_at`, `frames_scored`, `peak` | For the summary row |

**Raw per-frame scores are never shown.** They flicker hard — a single frame with motion blur or a bad crop swings the number — so the badge reads from `ema`.

**Hysteresis on the thresholds.** Entering SUSPICIOUS at `RISK_SUSPICIOUS` (0.4) but leaving only below 0.3, and entering HIGH_RISK at `RISK_HIGH` (0.7) but leaving only below 0.6. Without the gap, a score sitting on a threshold makes the badge strobe between two states several times a second. The entry thresholds stay the configured ones, so live and upload agree on what the numbers mean; only the exit is sticky.

**No face holds the last verdict.** A face that steps out of frame for a
moment must not read as REAL, so a frame with no detection leaves the badge
where it was and only reports `face_found: false`.

Sessions idle for more than a few minutes get reaped, so an abandoned tab cannot leak memory. Because state is process-local, a multi-instance Cloud Run deployment needs session affinity enabled — or, if that proves awkward, the client can carry the ring buffer and send it back, making the server stateless.

---

## 8. Persistence

A second table rather than overloading `analyses` — a live session has no filename, no storage path, and a duration instead of a suspicious region:

```sql
CREATE TABLE IF NOT EXISTS live_sessions (
    id                SERIAL PRIMARY KEY,
    started_at        TIMESTAMPTZ NOT NULL,
    ended_at          TIMESTAMPTZ NOT NULL,
    duration_seconds  DOUBLE PRECISION NOT NULL,
    frames_scored     INTEGER NOT NULL,
    mean_probability  DOUBLE PRECISION NOT NULL,
    peak_probability  DOUBLE PRECISION NOT NULL,
    status            TEXT NOT NULL,
    timeline          JSONB NOT NULL,
    user_feedback     TEXT
)
```

Created by `init_schema` alongside the existing table, same `CREATE TABLE IF NOT EXISTS` pattern.

**Written once, on stop.** Per-frame inserts at 2 fps would put thousands of round trips to Neon in the hot path for no benefit. `timeline` holds the same `{timestamp, fake_probability}` shape as `analyses.frame_scores`, so the history chart renders both with one code path.

`GET /history` grows a `kind` field (`"upload"` or `"live"`) and unions the two
tables, ordered by time. Ids are only unique within a kind, so anything that
follows a history row has to carry the kind with it. `GET /live/sessions/{id}`
reads one finished session back, and the Streamlit app gained a History tab
that lists both kinds and charts either timeline through the same code path.

**Written only if a face was seen.** A share that never scored a face has
nothing worth a row: `/stop` answers 422 and the client shows no summary.

---

## 9. Performance budget

MTCNN plus the classifier is roughly 150–400 ms per frame on CPU. That is the whole constraint, and everything here is chosen against it:

| Knob | Setting | Why |
|---|---|---|
| Sample rate | 2 fps | A face does not become fake between frames; 30 fps buys nothing and costs 15× |
| Capture width | 640 px | MTCNN's cost scales with pixels; faces on a shared screen are small but not *that* small |
| Detection cadence | MTCNN every 5th frame, reuse `last_box` in between | Detection dominates the per-frame cost, and a face does not teleport in 500 ms |
| JPEG quality | **0.95** | ~57 KB/frame, so ~114 KB/s uplink — see the note below |
| Inference | One worker, single-slot queue | See §6 |

Expected throughput on the `--cpu 2` Cloud Run sizing: 1–2 verdicts per second. `docs/deployment.md` measures 56 frames in 2.7 s on 16 local cores, which is consistent with this once the core count is scaled down.

**Measured during T3.** Sharing a looping known fake at 640 px / quality 0.95
on the 16-core dev machine holds 1.7–2.0 verdicts per second — the 500 ms tick
is the ceiling and `inFlight` drops the occasional tick, which is the intended
behaviour rather than a shortfall.

**Measured during T4 — box reuse is worth 2.3×.** Posting 640 px frames back
to back over HTTP on the same machine, with nothing else competing:

| Path | Per frame | Throughput |
|---|---|---|
| Session, MTCNN every 5th frame | 71 ms | 14.1 verdicts/s |
| Stateless, MTCNN every frame | 163 ms | 6.1 verdicts/s |

Detection really is the cost, as §9 assumed. Both numbers are far above the
2 fps the capture loop asks for, so the 500 ms tick stays the ceiling even
once the core count is scaled down to the Cloud Run sizing. Driven through the
browser instead — Chrome painting the capture canvas on the same machine — the
same run holds about 1 verdict/second, which is the floor the exit criterion
asks for and is bounded by the capture side, not by scoring.

**Measured during T2 — JPEG quality is not a free knob.** The classifier is
strongly sensitive to JPEG compression artifacts, far more than to resolution.
Re-encoding the peak frame of a known fake (whole-video verdict 0.9733
HIGH_RISK) at 640 px and posting it to `POST /live/frame`:

| Quality | 0.60 | 0.70 | 0.75 | 0.80 | 0.85 | 0.90 | 0.95 |
|---|---|---|---|---|---|---|---|
| Score | 0.001 | 0.020 | 0.001 | 0.103 | 0.478 | 0.900 | 0.995 |

The originally planned 0.7 reads a known fake as **REAL** — a false negative,
the worst direction for this app. The numbers are deterministic on repeat, and
monotonic in quality above 0.75. Resolution matters much less: at quality 0.95,
every width from 1280 down to 640 px holds HIGH_RISK (0.99+), while 480 px fails
even at 0.95 (crop falls to ~116×142). So 640 px stays, quality moves to 0.95,
and the uplink still lands inside the budget above.

This is a domain gap — the model was trained on frames decoded from video, not
on re-compressed stills — so the durable fix is to include JPEG-recompressed
augmentation in training. Until then the capture path must not compress hard,
and any future change to the capture quality needs this measurement re-run.

If `last_box` drift becomes visible — the crop sliding off the face during movement — the fix is to re-detect when the crop's score drops sharply, not to raise the detection cadence globally.

---

## 10. Privacy

Screen frames deserve more care than uploaded clips. An uploaded file is something the user chose and can see; a shared screen can sweep up an inbox, a chat window, a password manager, or a colleague's face — none of which the user is thinking about when they click Share.

Requirements, not suggestions:

- **An explicit consent line before capture starts**, naming what leaves the machine: still frames of the shared surface, sent to the backend, scored, and discarded.
- **Frames are held in memory only.** Nothing from a live session is written to `UPLOAD_DIR`, unlike the upload path which persists the file it analysed.
- **The stored timeline holds scores, not images.** No `storage_path` column on `live_sessions`, and no most-suspicious-frame thumbnail for live runs — that feature stays upload-only.
- **Prefer tab capture over full-screen** in the picker guidance shown to the user; it narrows what can be captured by accident.
- Frame bytes should not reach the application log at any level, including on the error paths.

---

## 11. Phased plan

| # | Task | Done when |
|---|---|---|
| T1 | Split `detect_face` / `score_frame` / `aggregate` out of `process_video` | `backend/smoke_test.py` passes unchanged |
| T2 | `POST /live/frame`, stateless, one JPEG in and one score out | A curl'd still of a known fake scores above 0.5 |
| T3 | Screen-share component, HTTP transport, live badge | Sharing a playing video moves the badge |
| T4 | `LiveSession`, EMA, hysteresis, 60 s sparkline | Badge holds steady on borderline content — **done**, see below |
| T5 | `live_sessions` table, write on stop, history union | A finished session appears in history — **done** |
| T6 | WebSocket transport | Only if T3–T5 latency measurements justify it — **not justified**, see §9 |

**T4 verification.** Alternating frames from a known real and a known fake
clip into one session, 24 frames: the raw score swings between 0.0000 and
0.7804 while the badge never changes state once. The threshold walk is covered
separately — 0.45 → 0.35 → 0.31 holds SUSPICIOUS, 0.29 drops to REAL; 0.72 →
0.65 → 0.61 holds HIGH_RISK, 0.55 falls back to SUSPICIOUS.

**T6 is not justified.** §9's T4 numbers put per-frame HTTP overhead nowhere
near the constraint — scoring is 71 ms and the client only asks every 500 ms —
so a WebSocket would buy nothing but a second transport to keep working.

T1 and T2 are backend-only and independently verifiable. T3 is the risky one — it is the first custom Streamlit component in this repo — so it gets its own step rather than being folded into the session work.

---

## 12. Alternatives and open questions

**In-browser inference.** Export the checkpoint to ONNX and run it with ONNX Runtime Web, pairing it with BlazeFace for detection. No frames leave the machine, no bandwidth, no per-frame Cloud Run cost, and §10 mostly stops being a concern. The cost is a second inference path to keep honest against the first, a quantised model with its own accuracy profile, and a much larger first-load. Worth revisiting once the server-side flow is working and we know what accuracy we are trading against.

**Open questions**

- Cloud Run session affinity for §7's in-process state, or push the ring buffer to the client and keep the server stateless?
- Does the live badge need the most-suspicious-frame thumbnail the upload flow shows? §10 argues against storing one; an in-browser-only version, never sent anywhere, would be a middle path.
- Should a live session with no face detected for N seconds surface "no face on screen" as a distinct state? T4 holds the last verdict instead and shows NO FACE only before the first face is ever seen, on the grounds that a face leaving frame for a moment must not read as REAL. A time limit on holding is still unanswered.
