#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/../shared/common_env.sh"
resolve_python

exec "$PY" "$SHARED_DIR/quiz_app.py" \
  --data_path "$ROOT_DIR/data/test.jsonl" \
  --port "${PORT:-7870}"
