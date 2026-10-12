#!/usr/bin/env bash
#
# Run the app with one command. The FastAPI backend also serves the pages in
# backend/static, so there is a single server.
#
#   ./run.sh                          # app on 8000
#   ./run.sh --port 8010
#   ./run.sh --skip-install           # never touch .venv
#
# A busy port is stepped past automatically. Ctrl-C stops the server.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PORT=8000
SKIP_INSTALL=0
LOG_DIR="$ROOT/logs"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --skip-install) SKIP_INSTALL=1; shift ;;
    -h|--help) sed -n '3,10p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1 (try --help)" >&2; exit 2 ;;
  esac
done

say() { printf '\033[1m▸ %s\033[0m\n' "$*"; }
die() { printf '\033[1m✗ %s\033[0m\n' "$*" >&2; exit 1; }

port_busy() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null && exec 3<&- && return 0 || return 1; }

free_port() {
  local port="$1"
  for _ in $(seq 20); do
    port_busy "$port" || { echo "$port"; return; }
    port=$((port + 1))
  done
  die "no free port in ${1}-$((${1} + 20))"
}

WANTED_PORT=$PORT
PORT="$(free_port "$PORT")"
[[ "$PORT" == "$WANTED_PORT" ]] || say "port $WANTED_PORT is taken — the app moves to $PORT"

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

if [[ ! -f backend/.env ]]; then
  say "backend/.env not found — copying backend/.env.example (set NEON_DB_URL in it)"
  cp backend/.env.example backend/.env
fi

mkdir -p "$LOG_DIR"

# Clerk tokens are only accepted from the origin the app is served on.
export CORS_ORIGINS="http://localhost:$PORT,http://127.0.0.1:$PORT"

BACKEND_PID=""

cleanup() {
  trap - INT TERM EXIT
  say "stopping"
  [[ -n "$BACKEND_PID" ]] && kill "$BACKEND_PID" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup INT TERM EXIT

say "starting the app on :$PORT"
"$PYTHON" -m uvicorn backend.main:app --port "$PORT" \
  > >(tee -a "$LOG_DIR/backend.log" | sed -u 's/^/\x1b[2m[api]\x1b[0m /') 2>&1 &
BACKEND_PID=$!

say "waiting for the model to load"
for i in $(seq 120); do
  kill -0 "$BACKEND_PID" 2>/dev/null || die "backend exited — see $LOG_DIR/backend.log"
  HEALTH="$(curl -sf "http://127.0.0.1:$PORT/health" || true)"
  [[ -n "$HEALTH" ]] && break
  sleep 1
done
[[ -n "${HEALTH:-}" ]] || die "backend did not answer /health within 120s — see $LOG_DIR/backend.log"

case "$HEALTH" in
  *'"model_loaded":true'*) say "checkpoint loaded" ;;
  *) say "no checkpoint found — the backend is in stub mode and scores every frame 0.5" ;;
esac

cat <<BANNER

  App        http://localhost:$PORT
  API docs   http://localhost:$PORT/docs
  Logs       $LOG_DIR/backend.log

  Ctrl-C stops the server.

BANNER

wait "$BACKEND_PID"
