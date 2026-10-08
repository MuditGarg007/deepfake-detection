# FastAPI backend + the v1 EfficientNet-B0 checkpoint, CPU-only.
# Build context is the repo root (see ../.dockerignore).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY backend/requirements-deploy.txt backend/requirements-deploy.txt
RUN pip install -r backend/requirements-deploy.txt \
 && pip install --no-deps "facenet-pytorch>=2.6"

COPY machine-learning/ machine-learning/
COPY backend/ backend/

ENV MODEL_DIR=machine-learning/checkpoints/efficientnet_b0_20260901_204509

EXPOSE 8000

# Run from the repo root: the detector imports machine-learning/inference.py by
# a root-relative path. One worker only — live sessions are process-local.
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
