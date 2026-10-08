# Frontend

Next.js UI over the FastAPI backend. One page, three tabs:

- **Screen share** — share a window or tab; a frame is captured every 500 ms and
  scored live through `/live/{session}/frame`. Stopping the share saves the
  session (`/live/{session}/stop`) and shows its average, peak and timeline.
- **Upload video** — drop an MP4, AVI or MOV (up to 200 MB) and get the verdict,
  the most suspicious frame, a per-frame timeline and the suspicious window.
  You can mark the video as real or fake; if that disagrees with the verdict the
  video is scored again.
- **History** — every upload and screen share, newest first. Click one to open it.

## Run

From the repository root, `./run.sh` starts the backend and this app together
(backend on :8000, frontend on :3000) and points one at the other.

On its own, with the backend already up:

```bash
npm install
npm run dev
```

`NEXT_PUBLIC_API_URL` in `.env.local` sets the backend address and defaults to
`http://localhost:8000`. The backend's `CORS_ORIGINS` has to include the origin
this app is served from.

## Layout

- `lib/api.ts` — every backend call; `lib/types.ts` mirrors `backend/schemas.py`
- `components/detector.tsx` — the page and its tabs
- `components/screen-share.tsx` — capture loop and live verdict
- `components/upload-card.tsx`, `components/result-panel.tsx` — upload flow
- `components/history-panel.tsx`, `components/live-summary.tsx` — history
- `components/timeline.tsx` — the probability chart
