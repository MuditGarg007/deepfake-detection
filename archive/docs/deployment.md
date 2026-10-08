# Deployment

The app is split across three tiers:

| Piece | Host | Why |
| --- | --- | --- |
| `backend/` + `machine-learning/` | Google Cloud Run | 2 GiB fits the model, and scale-to-zero keeps it free |
| `frontend/` | Vercel | Next.js, zero-config |
| Postgres | Neon | Cloud Run's filesystem is ephemeral, so SQLite would not survive |

The browser talks to Cloud Run directly (`NEXT_PUBLIC_API_URL`), so the
service's CORS list has to name the Vercel origin. Nothing is proxied through
Vercel.

## Why not Hugging Face Spaces

A compute Space is **not free**. Per the Hub docs: "Static Spaces are free for
everyone. Gradio and Docker Spaces run on compute and require a paid plan to
create: PRO for personal accounts, Team or Enterprise for organizations." The
"CPU Basic — FREE" row in the pricing table is the *hardware* rate; creating a
compute Space is gated separately.

Render's free tier is not an option: it caps at 512 MB, and this service
measures ~605 MB RSS with the checkpoint loaded.

## Sizing

Measured by running the service locally:

- ~605 MB RSS with the checkpoint loaded and one analysis done.
- An 11-second clip scored 56 frames in 2.7 s on 16 cores.

So `--memory 2Gi --cpu 2` leaves real headroom without wasting allowance.

## Free-tier budget

Cloud Run's always-free allowance is 240,000 vCPU-seconds, 450,000
GiB-seconds, and 2 million requests per month, and the default request-based
billing charges "only when they process requests, when they start, and when
they shut down" — idle instances between requests are not billed.

At `--cpu 2`, a 30-second analysis costs 60 vCPU-seconds, so the allowance is
roughly 4,000 analyses a month. The keep-warm ping below costs about 430
vCPU-seconds a month, under 0.2%.

`--max-instances 3` is the guard against a runaway bill; `--concurrency 2`
keeps one instance from serving more simultaneous analyses than its memory can
hold.

## One-time setup

### 1. Database

Create a free project at <https://neon.tech> and copy the connection string.
It is passed to psycopg2 as-is:

```
postgresql://USER:PASSWORD@HOST/dbname?sslmode=require
```

A `postgresql+psycopg2://` string is accepted too — the dialect suffix is
stripped before the connection is opened.

`backend/config.py` accepts it as either `NEON_DB_URL` or `DATABASE_URL`.

### 2. Git LFS and the checkpoint

`.gitignore` un-ignores only `efficientnet_b0_20260901_204509/`, which is what
`MODEL_DIR` points at. Without it the backend boots in stub mode and scores
every frame 0.5.

```bash
git lfs install
git add .gitattributes machine-learning/checkpoints/efficientnet_b0_20260901_204509
```

### 3. Deploy the backend

```bash
gcloud auth login
gcloud config set project <PROJECT_ID>
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com

gcloud run deploy deepfake-api \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --concurrency 2 \
  --max-instances 3 \
  --set-env-vars "MAX_UPLOAD_MB=50,CORS_ORIGINS=https://<your-app>.vercel.app" \
  --set-env-vars "DATABASE_URL=postgresql+psycopg2://..."
```

`--source .` builds with Cloud Build. The repo no longer carries a container
build of its own, so Cloud Build falls back to its Python buildpack — set
`MODEL_DIR` and `UPLOAD_DIR` explicitly with `--set-env-vars`, and confirm the
buildpack picks up `backend/requirements-deploy.txt` (the CPU-only torch
wheels) rather than a CUDA build.

For a secret you would rather not have in shell history, put it in Secret
Manager and swap the last line for
`--set-secrets "DATABASE_URL=deepfake-db-url:latest"`.

Verify:

```bash
curl https://<service-url>/health
# {"status":"ok","model_loaded":true}
```

`model_loaded: false` means an import failed or the checkpoint is missing, and
every frame would score a flat 0.5 — check step 2.

### 4. Keep-warm ping

Cloud Run scales to zero, and a cold request pays for the container start, the
torch import, and loading the checkpoint. A scheduled ping to `/health` keeps
an instance warm for a negligible slice of the free tier.

```bash
gcloud services enable cloudscheduler.googleapis.com

gcloud scheduler jobs create http deepfake-api-keepwarm \
  --location us-central1 \
  --schedule "*/5 * * * *" \
  --uri "https://<service-url>/health" \
  --http-method GET \
  --attempt-deadline 30s
```

This is best-effort, not a guarantee: Cloud Run may still reclaim the instance,
and the ping only makes a cold start less likely. The guaranteed alternative,
`--min-instances 1`, switches the service to billed idle time — an always-on
instance is roughly 2.6M vCPU-seconds a month against a 240,000 free
allowance, so it is not free.

Cloud Scheduler's free tier covers a small number of jobs per billing account;
one job is well inside it. Confirm the current figure at
<https://cloud.google.com/scheduler/pricing>.

### 5. Frontend

Import the repo at <https://vercel.com/new> with **Root Directory =
`frontend`**. Set one environment variable:

```
NEXT_PUBLIC_API_URL = https://<service-url>
```

`NEXT_PUBLIC_*` is inlined at build time, so changing it later needs a
redeploy, not just a restart.

### 6. Close the loop

Update `CORS_ORIGINS` to the domain Vercel actually assigned:

```bash
gcloud run services update deepfake-api --region us-central1 \
  --update-env-vars "CORS_ORIGINS=https://<your-app>.vercel.app"
```

Preview deployments get their own subdomains and will fail CORS unless added
too.

## What the deployment files do

- `backend/requirements-deploy.txt` — the same dependencies as
  `backend/requirements.txt` but resolved against PyTorch's CPU wheel index.
  The default PyPI `torch` bundles CUDA and drags in ~2.5 GB of `nvidia-*`
  packages a CPU host cannot use; the CPU wheel is 178 MB. It also adds
  `pandas`, which nothing in `backend/` imports but `machine-learning/
  dataset.py` does at module level — `inference.py` reaches it through
  `build_transform`, and without it the detector falls back to stub mode.
- No system `ffmpeg`: `opencv-python-headless` bundles its own, and installing
  it added ~150 packages for nothing.
- `.gitattributes` — routes `*.pth` through Git LFS.

The app must be started from the repo root — the detector imports
`machine-learning/inference.py` by a path relative to the project root — and
the command has to honor `$PORT` (Cloud Run injects 8080):

```bash
uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}
```

## Operating notes

- **Cold starts.** Scaling to zero means the first request after an idle
  period pays for the container start plus loading the checkpoint. The
  keep-warm ping above is the mitigation.
- **Ephemeral disk.** Uploaded videos live in `UPLOAD_DIR` inside the
  container and are gone when the instance is replaced. Analysis rows survive
  in Neon; `Analysis.has_video` turns false and the UI drops the player.
  Keeping clips across restarts needs an object store such as GCS.
- **Speed.** Inference is CPU-only, and `video_processor` samples up to
  `MAX_FRAMES = 100` frames, held open by the synchronous `POST /analyze`.
  Budget tens of seconds per clip at `--cpu 2`. Lowering `MAX_FRAMES` to 50
  halves it at some cost in coverage.
- **Open API.** `--allow-unauthenticated` means anyone can post videos and
  consume the allowance. `MAX_UPLOAD_MB` and `--max-instances` are the only
  limits in place.

## Testing the deployment build locally

```bash
python -m venv .venv-deploy && . .venv-deploy/bin/activate
pip install -r backend/requirements-deploy.txt
pip install --no-deps "facenet-pytorch>=2.6"
MODEL_DIR=machine-learning/checkpoints/efficientnet_b0_20260901_204509 \
    uvicorn backend.main:app --port 8000
curl localhost:8000/health
curl -X POST localhost:8000/analyze -F "file=@some-clip.mp4"
```

With no `DATABASE_URL` it falls back to SQLite, which is enough to check that
the model loads and the routes answer.
