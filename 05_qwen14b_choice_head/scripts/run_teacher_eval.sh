#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/../shared/common_env.sh"
resolve_python
resolve_model_dir BASE_MODEL_14B Qwen2.5-14B-Instruct

mkdir -p "$ROOT_DIR/logs"

"$PY" "$SHARED_DIR/generate_teacher_labels_local.py" \
  --model_path "$BASE_MODEL_14B" \
  --dataset "$ROOT_DIR/data/test.jsonl" \
  --output "$ROOT_DIR/data/teacher_test.jsonl" \
  2>&1 | tee "$ROOT_DIR/logs/teacher_test.log"

"$PY" "$SHARED_DIR/teacher_eval_summary.py" \
  --input "$ROOT_DIR/data/teacher_test.jsonl" \
  --report "$ROOT_DIR/logs/teacher_eval_report.json" \
  2>&1 | tee "$ROOT_DIR/logs/teacher_eval_summary.log"