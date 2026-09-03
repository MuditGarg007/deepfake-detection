# Deployment

The app is split across three free tiers:

| Piece | Host | Why |
| --- | --- | --- |
| `backend/` + `machine-learning/` | Hugging Face Space (Docker SDK) | 2 vCPU / 16 GB RAM free, enough for CPU torch + MTCNN |
| `frontend/` | Vercel | Next.js, zero-config |
| Postgres | Neon | the Space filesystem is ephemeral, so SQLite would not survive a restart |

The browser talks to the Space directly (`NEXT_PUBLIC_API_URL`), so the Space's
CORS list has to name the Vercel origin. Nothing is proxied through Vercel.

## What the deployment files do

- `Dockerfile` — builds the Space image from the repo root (the detector
  imports `machine-learning/inference.py` by a path relative to the project
  root, so `backend/` alone is not a valid build context).
- `backend/requirements-deploy.txt` — the same dependencies as
  `backend/requirements.txt` but resolved against PyTorch's CPU wheel index.
  The default PyPI `torch` bundles CUDA and drags in ~2.5 GB of `nvidia-*`
  packages a CPU Space cannot use; the CPU wheel is 178 MB. It also adds
  `pandas`, which nothing in `backend/` imports but `machine-learning/
  dataset.py` does at module level — `inference.py` reaches it through
  `build_transform`, and without it the detector falls back to stub mode.
- `.dockerignore` — keeps the frontend, the datasets, the Xception checkpoint,
  and local secrets out of the image.
- `README.md` front matter — how a Space is configured (`sdk: docker`,
  `app_port: 7860`). It has to live at the repo root.
- `.gitattributes` — routes `*.pth` through Git LFS. Hugging Face rejects
  non-LFS files over 10 MB and builds the Space from git, so the 17 MB
  checkpoint has no other way in.

## One-time setup

### 1. Database

Create a free project at <https://neon.tech> and copy the connection string.
Rewrite the scheme for SQLAlchemy's psycopg2 driver:

```
postgresql+psycopg2://USER:PASSWORD@HOST/dbname?sslmode=require
```

`backend/config.py` accepts it as either `NEON_DB_URL` or `DATABASE_URL`.

### 2. Git LFS and the checkpoint

`.gitignore` un-ignores only `efficientnet_b0_20260901_204509/`, which is what
`MODEL_DIR` points at. Without it the backend boots in stub mode and scores
every frame 0.5.

```bash
sudo pacman -S git-lfs      # or your platform's package
git lfs install
git add .gitattributes machine-learning/checkpoints/efficientnet_b0_20260901_204509
git commit -m "chore: ship the served checkpoint via LFS"
```

### 3. Create the Space

At <https://huggingface.co/new-space>: **SDK → Docker → Blank**, visibility
**public**. A private Space requires a bearer token on every request, which a
browser client has nowhere to keep.

```bash
git remote add hf https://huggingface.co/spaces/<user>/<space>
git push hf main
```

The first build takes 10-15 minutes, almost all of it installing torch. Later
builds reuse that layer as long as `requirements-deploy.txt` is untouched.

### 4. Configure the Space

Under **Settings → Variables and secrets**:

| Name | Kind | Value |
| --- | --- | --- |
| `DATABASE_URL` | secret | the Neon string from step 1 |
| `CORS_ORIGINS` | variable | `https://<your-app>.vercel.app` |
| `MAX_UPLOAD_MB` | variable | `50` |

Both kinds arrive as environment variables, which beat `backend/.env` in
pydantic-settings. Changing either needs a rebuild ("Factory rebuild" is only
required if the image itself is stale).

Check it came up:

```bash
curl https://<user>-<space>.hf.space/health
# {"status":"ok","model_loaded":true}
```

`model_loaded: false` means the checkpoint did not make it into the image —
check step 2. Note the API origin is `<user>-<space>.hf.space`; the
`huggingface.co/spaces/...` URL is only an iframe wrapper around it.

### 5. Frontend

Import the repo at <https://vercel.com/new> with **Root Directory =
`frontend`**. Set one environment variable:

```
NEXT_PUBLIC_API_URL = https://<user>-<space>.hf.space
```

`NEXT_PUBLIC_*` is inlined at build time, so changing it later needs a
redeploy, not just a restart.

### 6. Close the loop

Set `CORS_ORIGINS` on the Space to the domain Vercel actually assigned and
rebuild. Preview deployments get their own subdomains and will fail CORS unless
they are added too.

## Operating notes

- **Sleep.** A free Space sleeps after 48 hours idle. The next request wakes it
  in 1-2 minutes: a container restart, plus loading the checkpoint.
- **Ephemeral disk.** Uploaded videos live in `UPLOAD_DIR` inside the
  container and are gone after any restart or rebuild. Analysis rows survive in
  Neon; `Analysis.has_video` turns false and the UI drops the player. Keeping
  clips across restarts needs paid persistent storage, or an object store.
- **Speed.** Inference is CPU-only. `video_processor` samples up to
  `MAX_FRAMES = 100` frames and runs MTCNN on each, held open by the
  synchronous `POST /analyze`. Measured locally on 16 cores: an 11-second clip
  scored 56 frames in 2.7 s. A Space has 2 vCPU, so budget roughly 20-60 s for
  a clip that size and more for longer ones. Lowering `MAX_FRAMES` to 50 halves
  it at some cost in coverage.
- **Memory.** The container sits at ~605 MB RSS with the checkpoint loaded and
  one analysis done — comfortable against the Space's 16 GB, but the reason a
  512 MB free tier elsewhere is not an option.
- **Open API.** A public Space is an unauthenticated endpoint — anyone can post
  videos and consume the CPU. `MAX_UPLOAD_MB` is the only limit in place.

## Testing the image locally

```bash
docker build -t deepfake-api .
docker run --rm -p 7860:7860 deepfake-api
curl localhost:7860/health
```

With no `DATABASE_URL` it falls back to SQLite inside the container, which is
enough to check that the model loads and the routes answer. `model_loaded`
must be `true`; `false` means an import failed or the checkpoint is missing,
and every frame would score a flat 0.5. A full check:

```bash
curl -X POST localhost:7860/analyze -F "file=@some-clip.mp4"
```

The image is ~420 MB compressed, ~1.8 GB unpacked.
