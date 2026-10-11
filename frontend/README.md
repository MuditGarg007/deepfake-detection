# Frontend

Next.js UI over the FastAPI backend. One page, three tabs:

- **Screen share** — share a window or tab; a frame is captured every 500 ms and
  scored live through `/live/{session}/frame`. Stopping the share saves the
  session (`/live/{session}/stop`) and shows its average, peak and timeline.
- **Upload video** — drop an MP4, AVI or MOV (up to 200 MB) and get the verdict,
  the most suspicious frame, a per-frame timeline and the suspicious window.
  You can mark the video as real or fake; if that disagrees with the verdict the
  video is scored again.
- **History** — your uploads and screen shares, newest first. Click one to open it.

Everything sits behind Clerk sign-in. `/sign-in` and `/sign-up` are custom forms
over Clerk's hooks: email + password (new accounts confirm their address with an
emailed 6-digit code) or Google / GitHub. `/sso-callback` finishes the OAuth
redirect. Every backend call carries the Clerk session token.

## Run

From the repository root, `./run.sh` starts the backend and this app together
(backend on :8000, frontend on :3000) and points one at the other.

On its own, with the backend already up:

```bash
npm install
npm run dev
```

Copy `.env.example` to `.env.local` and fill in the Clerk keys.
`NEXT_PUBLIC_API_URL` sets the backend address and defaults to
`http://localhost:8000`. The backend's `CORS_ORIGINS` has to include the origin
this app is served from.

## Layout

- `lib/api.ts` — every backend call; `lib/types.ts` mirrors `backend/schemas.py`
- `components/detector.tsx` — the page and its tabs
- `components/auth.tsx`, `components/sign-in-form.tsx`, `components/sign-up-form.tsx` — sign-in and sign-up
- `components/screen-share.tsx` — capture loop and live verdict
- `components/upload-card.tsx`, `components/result-panel.tsx` — upload flow
- `components/history-panel.tsx`, `components/live-summary.tsx` — history
- `components/timeline.tsx` — the probability chart
