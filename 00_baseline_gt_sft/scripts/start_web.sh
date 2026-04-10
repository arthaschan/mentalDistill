#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/../shared/common_env.sh"
resolve_python
resolve_model_dir BASE_MODEL_7B Qwen2.5-7B-Instruct

ADAPTER_DIR="${ADAPTER_DIR:-$ROOT_DIR/outputs/best}"
if [[ ! -d "$ADAPTER_DIR" ]]; then
  echo "未找到 adapter 目录: $ADAPTER_DIR，请先训练或通过 ADAPTER_DIR 指定。"
  exit 1
fi

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-7860}"

exec "$PY" "$SHARED_DIR/serve_model_app.py" \
  --base_model "$BASE_MODEL_7B" \
  --adapter_dir "$ADAPTER_DIR" \
  --adapter_root "$ROOT_DIR/outputs" \
  --host "$HOST" \
  --port "$PORT" \
  --title "Baseline GT SFT"