#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/../shared/common_env.sh"
resolve_python
resolve_model_dir BASE_MODEL_7B Qwen2.5-7B-Instruct

ADAPTER_DIR="${ADAPTER_DIR:-}"
ADAPTER_ROOT="${ADAPTER_ROOT:-$ROOT_DIR/runs}"

exec "$PY" "$SHARED_DIR/serve_model_app.py" \
  --base_model "$BASE_MODEL_7B" \
  --adapter_dir "$ADAPTER_DIR" \
  --adapter_root "$ADAPTER_ROOT" \
  --host "${HOST:-0.0.0.0}" \
  --port "${PORT:-7860}" \
  --title "Module 17 α-divergence 蒸馏"
