#!/usr/bin/env bash

# Start/stop/status helper for the SplitOculo remote demo.
#
# Usage:
#   ./scripts/start_demo_service.sh              # start
#   ./scripts/start_demo_service.sh status
#   ./scripts/start_demo_service.sh stop
#
# The cloud server and ngrok must run in the same container/host for the
# default 127.0.0.1:8080 target to work.

set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv-temporal/bin/python}"
CLOUD_HOST="${CLOUD_HOST:-127.0.0.1}"
CLOUD_PORT="${CLOUD_PORT:-8080}"
QWEN_PATH="${QWEN_PATH:-Qwen/Qwen2.5-VL-32B-Instruct}"
MAX_VIDEO_FRAMES="${MAX_VIDEO_FRAMES:-16}"
# Default to the spatial cloud half of the validated LLaVA-558K split model.
# Override with CLOUD_CHECKPOINT=/path/to/cloud_weights.pth for another run.
CLOUD_CHECKPOINT="${CLOUD_CHECKPOINT:-$ROOT_DIR/checkpoints/llava558k_32b_49x64_rolling/split/cloud_weights.pth}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"
CLOUD_LOG="${CLOUD_LOG:-$LOG_DIR/cloud_server.log}"
NGROK_LOG="${NGROK_LOG:-$LOG_DIR/ngrok.log}"
NGROK_API="${NGROK_API:-http://127.0.0.1:4040}"

# These defaults match the current demo. Override them without editing the
# script, for example:
#   NGROK_BASIC_AUTH_PASSWORD='my-password' ./scripts/start_demo_service.sh
NGROK_BASIC_AUTH_USER="${NGROK_BASIC_AUTH_USER:-splitoculo}"
NGROK_BASIC_AUTH_PASSWORD="${NGROK_BASIC_AUTH_PASSWORD:-SplitOculoDemo2026}"

CLOUD_PID_FILE="${CLOUD_PID_FILE:-/tmp/cloud_server.pid}"
NGROK_PID_FILE="${NGROK_PID_FILE:-/tmp/ngrok.pid}"
PUBLIC_URL_FILE="${PUBLIC_URL_FILE:-$LOG_DIR/public_url.txt}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "missing command: $1"
}

pid_value() {
  local pid_file="$1"
  [[ -s "$pid_file" ]] || return 1
  local pid
  pid="$(tr -dc '0-9' < "$pid_file")"
  [[ -n "$pid" ]] || return 1
  printf '%s\n' "$pid"
}

pid_is_alive() {
  local pid="$1"
  kill -0 "$pid" 2>/dev/null
}

pid_command() {
  local pid="$1"
  ps -p "$pid" -o args= 2>/dev/null || true
}

cloud_pid_is_ours() {
  local pid
  pid="$(pid_value "$CLOUD_PID_FILE" 2>/dev/null || true)"
  [[ -n "$pid" ]] || return 1
  pid_is_alive "$pid" || return 1
  pid_command "$pid" | grep -Fq "scripts/cloud_server.py"
}

ngrok_pid_is_ours() {
  local pid
  pid="$(pid_value "$NGROK_PID_FILE" 2>/dev/null || true)"
  [[ -n "$pid" ]] || return 1
  pid_is_alive "$pid" || return 1
  pid_command "$pid" | grep -Fq "ngrok http"
}

cloud_health() {
  curl -fsS --max-time 5 "http://$CLOUD_HOST:$CLOUD_PORT/health" 2>/dev/null
}

cloud_ready() {
  local response
  response="$(cloud_health || true)"
  [[ "$response" == *'"model_loaded":true'* ]] || return 1
  [[ "$response" == *'"qwen_loaded":true'* ]] || return 1
}

cloud_port_is_open() {
  (echo >/dev/tcp/127.0.0.1/"$CLOUD_PORT") >/dev/null 2>&1
}

public_url_from_ngrok() {
  curl -fsS --max-time 3 "$NGROK_API/api/tunnels" 2>/dev/null \
    | python3 -c '
import json
import sys

try:
    data = json.load(sys.stdin)
    urls = [item.get("public_url", "") for item in data.get("tunnels", [])]
    print(next((url for url in urls if url.startswith("https://")), ""))
except Exception:
    print("")
'
}

start_cloud() {
  [[ -x "$PYTHON_BIN" ]] || die "Python interpreter not executable: $PYTHON_BIN"
  [[ -f "$CLOUD_CHECKPOINT" ]] || die "cloud checkpoint not found: $CLOUD_CHECKPOINT"

  if cloud_ready; then
    echo "cloud server is already ready: http://$CLOUD_HOST:$CLOUD_PORT"
    return 0
  fi

  if cloud_pid_is_ours; then
    echo "cloud server is loading; pid=$(pid_value "$CLOUD_PID_FILE")"
  elif cloud_port_is_open; then
    die "port $CLOUD_PORT is already occupied, but it is not a ready SplitOculo cloud server; inspect with: ss -ltnp | grep :$CLOUD_PORT"
  else
    echo "starting cloud server with $QWEN_PATH ..."
    # Start in a fresh session so the cloud server survives the shell/tool
    # that launched this helper.  nohup alone does not detach the process
    # group in all container environments.
    setsid "$PYTHON_BIN" scripts/cloud_server.py \
      --checkpoint "$CLOUD_CHECKPOINT" \
      --host "$CLOUD_HOST" \
      --port "$CLOUD_PORT" \
      --device cuda \
      --qwen_path "$QWEN_PATH" \
      --max_video_frames "$MAX_VIDEO_FRAMES" \
      --offline \
      --preload_qwen \
      >"$CLOUD_LOG" 2>&1 < /dev/null &
    echo "$!" > "$CLOUD_PID_FILE"
    echo "cloud server pid=$(cat "$CLOUD_PID_FILE")"
  fi

  echo "waiting for cloud server and Qwen warmup ..."
  for _ in $(seq 1 600); do
    if cloud_ready; then
      echo "cloud server ready: http://$CLOUD_HOST:$CLOUD_PORT"
      return 0
    fi

    if ! cloud_pid_is_ours && ! cloud_port_is_open; then
      echo "cloud server exited before becoming ready; recent log:" >&2
      tail -n 80 "$CLOUD_LOG" >&2 || true
      return 1
    fi
    sleep 1
  done

  echo "cloud server warmup timed out; recent log:" >&2
  tail -n 80 "$CLOUD_LOG" >&2 || true
  return 1
}

start_ngrok() {
  require_command ngrok

  local existing_url
  existing_url="$(public_url_from_ngrok || true)"
  if [[ -n "$existing_url" ]]; then
    if ngrok_pid_is_ours; then
      echo "ngrok tunnel already exists: $existing_url"
      printf '%s\n' "$existing_url" > "$PUBLIC_URL_FILE"
      return 0
    fi
    die "an unmanaged ngrok tunnel already exists at $existing_url; stop it first, then rerun this script"
  fi

  echo "starting ngrok tunnel ..."
  setsid ngrok http "$CLOUD_HOST:$CLOUD_PORT" \
    --basic-auth "$NGROK_BASIC_AUTH_USER:$NGROK_BASIC_AUTH_PASSWORD" \
    --log=stdout \
    >"$NGROK_LOG" 2>&1 < /dev/null &
  echo "$!" > "$NGROK_PID_FILE"
  echo "ngrok pid=$(cat "$NGROK_PID_FILE")"

  local public_url=""
  for _ in $(seq 1 60); do
    public_url="$(public_url_from_ngrok || true)"
    if [[ -n "$public_url" ]]; then
      printf '%s\n' "$public_url" > "$PUBLIC_URL_FILE"
      echo "ngrok public URL: $public_url"
      return 0
    fi
    sleep 1
  done

  echo "ngrok did not publish a URL; recent log:" >&2
  tail -n 80 "$NGROK_LOG" >&2 || true
  return 1
}

verify_public_service() {
  local public_url
  public_url="$(cat "$PUBLIC_URL_FILE" 2>/dev/null || true)"
  [[ -n "$public_url" ]] || die "public URL file is empty: $PUBLIC_URL_FILE"

  local response
  response="$(curl -fsS --max-time 20 \
    -u "$NGROK_BASIC_AUTH_USER:$NGROK_BASIC_AUTH_PASSWORD" \
    "$public_url/health")" \
    || die "public health check failed: $public_url/health"

  [[ "$response" == *'"model_loaded":true'* ]] \
    || die "public endpoint reached, but cloud model is not ready: $response"
  [[ "$response" == *'"qwen_loaded":true'* ]] \
    || die "public endpoint reached, but Qwen is not ready: $response"

  echo "public service is ready"
  echo "PUBLIC_URL=$public_url"
  echo "BASIC_AUTH_USER=$NGROK_BASIC_AUTH_USER"
  echo "BASIC_AUTH_PASSWORD=$NGROK_BASIC_AUTH_PASSWORD"
  echo "cloud log: $CLOUD_LOG"
  echo "ngrok log: $NGROK_LOG"
}

start_service() {
  require_command curl
  require_command python3
  mkdir -p "$LOG_DIR"
  start_cloud
  start_ngrok
  verify_public_service
}

stop_pid_file() {
  local label="$1"
  local pid_file="$2"
  local expected="$3"
  local pid
  pid="$(pid_value "$pid_file" 2>/dev/null || true)"

  if [[ -z "$pid" ]] || ! pid_is_alive "$pid"; then
    echo "$label is not running"
    return 0
  fi

  if ! pid_command "$pid" | grep -Fq "$expected"; then
    echo "refusing to stop unexpected process pid=$pid ($label): $(pid_command "$pid")" >&2
    return 1
  fi

  kill "$pid"
  echo "stopped $label pid=$pid"
}

stop_service() {
  stop_pid_file ngrok "$NGROK_PID_FILE" "ngrok http" || true
  stop_pid_file cloud-server "$CLOUD_PID_FILE" "scripts/cloud_server.py" || true
}

status_service() {
  local active_url
  echo "project: $ROOT_DIR"
  echo "cloud pid: $(pid_value "$CLOUD_PID_FILE" 2>/dev/null || echo none)"
  echo "ngrok pid: $(pid_value "$NGROK_PID_FILE" 2>/dev/null || echo none)"
  echo "local health:"
  cloud_health || true
  echo
  echo "public URL file: $(cat "$PUBLIC_URL_FILE" 2>/dev/null || echo none)"
  active_url="$(public_url_from_ngrok || true)"
  if [[ -n "$active_url" ]]; then
    echo "ngrok tunnel: $active_url"
  else
    echo "ngrok tunnel: offline"
  fi
}

case "${1:-start}" in
  start)
    start_service
    ;;
  stop)
    stop_service
    ;;
  status)
    status_service
    ;;
  *)
    echo "usage: $0 [start|status|stop]" >&2
    exit 2
    ;;
esac
