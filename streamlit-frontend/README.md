# Streamlit frontend

A minimal alternative to the Next.js UI in `frontend/`: upload a video, run the
analysis, see the verdict. Both talk to the same FastAPI backend, so either one
can be used on its own.

## Run

```bash
pip install -r streamlit-frontend/requirements.txt
streamlit run streamlit-frontend/app.py
```

The backend must already be running (`uvicorn backend.main:app --port 8000`, or
`./run.sh`). The app defaults to `http://localhost:8000`; point it elsewhere
with `API_URL`:

```bash
API_URL=http://localhost:8010 streamlit run streamlit-frontend/app.py
```

`NEXT_PUBLIC_API_URL` is also read, so `./run.sh`'s exported value works too.

## What it shows

`POST /analyze` returns the whole analysis synchronously, so one request gives
the status (`REAL` / `SUSPICIOUS` / `HIGH_RISK`), the mean fake probability, the
suspicious time window, a chart of the per-frame probabilities, and the peak
frame from `GET /analysis/{id}/frame`.
