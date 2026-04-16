#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/../shared/common_env.sh"
resolve_python

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-7870}"

exec "$PY" "$SHARED_DIR/quiz_app.py" \
  --test_data "$ROOT_DIR/data/test.jsonl" \
  --host "$HOST" \
  --port "$PORT" \
  --title "$(basename "$ROOT_DIR") 人机测试"
