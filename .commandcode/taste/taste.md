# Taste

## Core principles
- Keep everything as beginner-friendly as possible — the user explicitly stated this is the project's "core principle." Prefer simple, readable code, plain-language comments explaining intent, and friendly error messages. Confidence: 0.95

## Project context (deepfake-detection)
- Stack: Python FastAPI backend, PostgreSQL, static HTML/CSS/JS frontend served from `backend/static`, Clerk for third-party auth. Runs via `run.sh` with a project-local `.venv`. Confidence: 0.85

## Architecture
- Auth flow: save user credentials into the project's own database (email + salted scrypt hash, never plaintext) before they go to Clerk — "store it as if the DB is the one handling auth," so the DB copy works standalone if Clerk is ever dropped. Confidence: 0.85
