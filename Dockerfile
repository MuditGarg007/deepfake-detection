# Hugging Face Space (Docker SDK) image for the FastAPI backend.
#
# The build context is the repo root, not backend/, because the detector
# imports machine-learning/inference.py through a path relative to the project
# root. See .dockerignore for what is left out.

FROM python:3.12-slim

# opencv-python-headless drops the GUI libraries but still links libglib. Its
# wheel bundles its own FFmpeg, so system ffmpeg is not needed to decode the
# uploaded mp4/mov/avi and would add ~150 packages to the image.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Spaces run the container as uid 1000, so build and run as that user rather
# than leaving root-owned files the app cannot write to.
RUN useradd -m -u 1000 user
USER user
ENV PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

# Requirements first: this layer is cached until the file itself changes, which
# matters because installing torch takes most of the build.
COPY --chown=user backend/requirements-deploy.txt backend/
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r backend/requirements-deploy.txt \
    && pip install --no-cache-dir --no-deps "facenet-pytorch>=2.6"

COPY --chown=user backend/ backend/
COPY --chown=user machine-learning/ machine-learning/

# UPLOAD_DIR is absolute so it lands next to the app rather than inside
# backend/; the Space filesystem is ephemeral either way, so uploads survive
# until the next restart or rebuild and Analysis.has_video reports false after.
ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/home/user/.cache/huggingface \
    UPLOAD_DIR=/home/user/app/uploads \
    MODEL_DIR=machine-learning/checkpoints/efficientnet_b0_20260901_204509

# 7860 is the Space default, matched by app_port in README.md's front matter.
EXPOSE 7860
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
