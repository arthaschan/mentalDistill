#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/../shared/common_env.sh"
resolve_python

STUDENT_SIZE="${STUDENT_SIZE:-14b}"
if [[ "$STUDENT_SIZE" == "14b" ]]; then
  resolve_model_dir BASE_MODEL_14B Qwen2.5-14B-Instruct
  BASE_MODEL="$BASE_MODEL_14B"
else
  resolve_model_dir BASE_MODEL_7B Qwen2.5-7B-Instruct
  BASE_MODEL="$BASE_MODEL_7B"
fi

ADAPTER_DIR="${ADAPTER_DIR:-}"
ADAPTER_ROOT="${ADAPTER_ROOT:-$ROOT_DIR/runs}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-7860}"

exec "$PY" "$SHARED_DIR/serve_model_app.py" \
  --base_model "$BASE_MODEL" \
  --adapter_dir "$ADAPTER_DIR" \
  --adapter_root "$ADAPTER_ROOT" \
  --host "$HOST" \
  --port "$PORT" \
  --title "Module 15 全量数据蒸馏 (${STUDENT_SIZE}B)"
