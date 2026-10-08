# Deepfake Detection & Alert System

Upload a video or share your screen, and get a frame-by-frame fake probability,
an overall risk status, and the most suspicious region of the clip.

- `machine-learning/` — training and inference for the EfficientNet-B0 /
  Xception classifiers (Phase 1). See `machine-learning/README.md`.
- `backend/` — FastAPI service wrapping the detector. See `backend/README.md`.
- `frontend/` — Next.js UI. See `frontend/README.md`.
- `archive/` — the retired Streamlit UI and older docs.
- `docs/` — the plan and the per-phase roadmaps.

## Running locally

```bash
./run.sh          # backend on :8000, frontend on :3000
```

## Deployment

The backend runs on Google Cloud Run and the frontend on Vercel.
`docs/deployment.md` has the full walkthrough.
