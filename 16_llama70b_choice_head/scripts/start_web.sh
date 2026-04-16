#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/../shared/common_env.sh"
resolve_python
resolve_model_dir BASE_MODEL_14B Qwen2.5-14B-Instruct

ADAPTER_DIR="${ADAPTER_DIR:-}"
ADAPTER_ROOT="${ADAPTER_ROOT:-$ROOT_DIR/runs}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-7860}"

exec "$PY" "$SHARED_DIR/serve_model_app.py" \
  --base_model "$BASE_MODEL_14B" \
  --adapter_dir "$ADAPTER_DIR" \
  --adapter_root "$ADAPTER_ROOT" \
  --host "$HOST" \
  --port "$PORT" \
  --title "Module 16 Llama-70B→14B 蒸馏"
