# Frontend — Deepfake Detection & Alert System

**Phase 5** of [`docs/plan.md`](../docs/plan.md): a Next.js App Router demo UI over the
Phase 4 FastAPI backend. No auth — upload a video, watch the pipeline run, read the
verdict, and browse past analyses.

## Run

Both halves at once, from the repository root:

```bash
./run.sh                    # backend :8000, frontend :3000 — Ctrl-C stops both
./run.sh --frontend-port 3010
```

It creates `.venv` / installs dependencies if they are missing, steps past busy
ports, and points the frontend at whichever port the backend actually got.

To run them separately, the backend must be up first (see [`backend/README.md`](../backend/README.md)):

```bash
uvicorn backend.main:app --reload --port 8000
```

Then:

```bash
npm install
npm run dev            # http://localhost:3000
```

`NEXT_PUBLIC_API_URL` (in `.env.local`) points the client at the backend and defaults
to `http://localhost:8000`. The backend's `CORS_ORIGINS` must contain the origin the
frontend is served from — it allows `localhost:3000` out of the box.

## Screens

Everything lives on `/`, driven by `components/detector.tsx`:

| Piece | Component | Endpoint |
|---|---|---|
| Upload + progress | `components/upload-card.tsx` | `POST /analyze` |
| Verdict, thresholds, suspicious window | `components/result-panel.tsx` | — |
| Per-frame chart + data table | `components/frame-timeline.tsx` | — |
| Recent analyses rail | `components/history-panel.tsx` | `GET /history`, `GET /analysis/{id}` |
| Backend/model state | `components/health-badge.tsx` | `GET /health` |

`lib/api.ts` is the only place that talks to the backend; `lib/types.ts` mirrors
`backend/schemas.py`. `POST /analyze` goes through `XMLHttpRequest` so the upload half
of the wait shows a real percentage — detection itself has no progress to report and
falls back to an indeterminate bar with an elapsed clock.

## Design

Light, monochrome, Geist. Risk state is carried by fill weight **and** a glyph **and**
the written label, never by shade alone, so the three states stay distinguishable in
grayscale and for color-vision deficiency. Risk thresholds (0.40 / 0.70) are mirrored
from `backend/config.py` in `lib/format.ts` — change them in both places.
