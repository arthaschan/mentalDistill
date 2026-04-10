#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/../shared/common_env.sh"
resolve_python
resolve_model_dir BASE_MODEL_7B Qwen2.5-7B-Instruct

cd "$ROOT_DIR"

"$PY" "$SHARED_DIR/evaluate_model.py" \
  --base_model "$BASE_MODEL_7B" \
  --adapter_dir "$ROOT_DIR/outputs/best" \
  --test_data "$ROOT_DIR/data/test.jsonl" \
  --wrong_log "$ROOT_DIR/outputs/test_wrong.jsonl"
