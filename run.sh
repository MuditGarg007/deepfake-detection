#!/usr/bin/env bash
#
# Run the whole app — Phase 4 backend + Phase 5 frontend — with one command.
#
#   ./run.sh                          # backend on 8000, frontend on 3000
#   ./run.sh --backend-port 8010 --frontend-port 3010
#   ./run.sh --skip-install           # never touch .venv / node_modules
#
# Busy ports are stepped past automatically, the frontend is pointed at whatever
# port the backend actually got, and Ctrl-C stops both.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

BACKEND_PORT=8000
FRONTEND_PORT=3000
SKIP_INSTALL=0
LOG_DIR="$ROOT/logs"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend-port) BACKEND_PORT="$2"; shift 2 ;;
    --frontend-port) FRONTEND_PORT="$2"; shift 2 ;;
    --skip-install) SKIP_INSTALL=1; shift ;;
    -h|--help) sed -n '3,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1 (try --help)" >&2; exit 2 ;;
  esac
done

say() { printf '\033[1m▸ %s\033[0m\n' "$*"; }
die() { printf '\033[1m✗ %s\033[0m\n' "$*" >&2; exit 1; }

# --- ports -------------------------------------------------------------------

port_busy() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null && exec 3<&- && return 0 || return 1; }

# First free port at or above $1, giving up after 20 tries.
free_port() {
  local port="$1"
  for _ in $(seq 20); do
    port_busy "$port" || { echo "$port"; return; }
    port=$((port + 1))
  done
  die "no free port in ${1}-$((${1} + 20))"
}

WANTED_BACKEND=$BACKEND_PORT
WANTED_FRONTEND=$FRONTEND_PORT
BACKEND_PORT="$(free_port "$BACKEND_PORT")"
FRONTEND_PORT="$(free_port "$FRONTEND_PORT")"
[[ "$BACKEND_PORT" == "$WANTED_BACKEND" ]] || say "port $WANTED_BACKEND is taken — backend moves to $BACKEND_PORT"
[[ "$FRONTEND_PORT" == "$WANTED_FRONTEND" ]] || say "port $WANTED_FRONTEND is taken — frontend moves to $FRONTEND_PORT"

# --- dependencies ------------------------------------------------------------

command -v npm >/dev/null || die "npm not found — install Node 18+ first"

if [[ ! -d .venv ]]; then
  [[ $SKIP_INSTALL == 1 ]] && die ".venv is missing and --skip-install was passed"
  say "creating .venv"
  python3 -m venv .venv
fi
PYTHON="$ROOT/.venv/bin/python"

if ! "$PYTHON" -c "import uvicorn, fastapi, cv2, torch" >/dev/null 2>&1; then
  [[ $SKIP_INSTALL == 1 ]] && die "backend dependencies are missing and --skip-install was passed"
  say "installing backend dependencies (this is slow the first time — torch)"
  "$PYTHON" -m pip install --quiet --upgrade pip
  "$PYTHON" -m pip install --quiet -r backend/requirements.txt
fi

if [[ ! -d frontend/node_modules ]]; then
  [[ $SKIP_INSTALL == 1 ]] && die "frontend/node_modules is missing and --skip-install was passed"
  say "installing frontend dependencies"
  (cd frontend && npm install --silent)
fi

if [[ ! -f backend/.env ]]; then
  say "backend/.env not found — copying backend/.env.example (SQLite fallback, no Neon)"
  cp backend/.env.example backend/.env
fi

mkdir -p "$LOG_DIR"

# --- start both --------------------------------------------------------------

# Env beats backend/.env in pydantic-settings, so CORS follows the real port.
export CORS_ORIGINS="http://localhost:$FRONTEND_PORT,http://127.0.0.1:$FRONTEND_PORT"
# Next inlines NEXT_PUBLIC_* from the environment, so no .env.local edit needed.
export NEXT_PUBLIC_API_URL="http://localhost:$BACKEND_PORT"

BACKEND_PID=""
FRONTEND_PID=""

cleanup() {
  trap - INT TERM EXIT
  say "stopping"
  [[ -n "$FRONTEND_PID" ]] && kill "$FRONTEND_PID" 2>/dev/null || true
  [[ -n "$BACKEND_PID" ]] && kill "$BACKEND_PID" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup INT TERM EXIT

say "starting backend on :$BACKEND_PORT"
"$PYTHON" -m uvicorn backend.main:app --port "$BACKEND_PORT" \
  > >(tee -a "$LOG_DIR/backend.log" | sed -u 's/^/\x1b[2m[api]\x1b[0m /') 2>&1 &
BACKEND_PID=$!

# The first boot loads torch and the checkpoint, so allow a generous wait.
say "waiting for the model to load"
for i in $(seq 120); do
  kill -0 "$BACKEND_PID" 2>/dev/null || die "backend exited — see $LOG_DIR/backend.log"
  HEALTH="$(curl -sf "http://127.0.0.1:$BACKEND_PORT/health" || true)"
  [[ -n "$HEALTH" ]] && break
  sleep 1
done
[[ -n "${HEALTH:-}" ]] || die "backend did not answer /health within 120s — see $LOG_DIR/backend.log"

case "$HEALTH" in
  *'"model_loaded":true'*) say "checkpoint loaded" ;;
  *) say "no checkpoint found — the backend is in stub mode and scores every frame 0.5" ;;
esac

say "starting frontend on :$FRONTEND_PORT"
(cd frontend && npm run dev -- --port "$FRONTEND_PORT") \
  > >(tee -a "$LOG_DIR/frontend.log" | sed -u 's/^/\x1b[2m[web]\x1b[0m /') 2>&1 &
FRONTEND_PID=$!

for i in $(seq 60); do
  kill -0 "$FRONTEND_PID" 2>/dev/null || die "frontend exited — see $LOG_DIR/frontend.log"
  curl -sf -o /dev/null "http://127.0.0.1:$FRONTEND_PORT" && break
  sleep 1
done

cat <<BANNER

  App        http://localhost:$FRONTEND_PORT
  API docs   http://localhost:$BACKEND_PORT/docs
  Logs       $LOG_DIR/{backend,frontend}.log

  Ctrl-C stops both.

BANNER

wait -n "$BACKEND_PID" "$FRONTEND_PID"
