#!/usr/bin/env bash
# Module 16: Llama-3.3-70B-Instruct 教师标签生成（本地真实 logprobs）
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/../shared/common_env.sh"
resolve_python

MODEL="${BASE_MODEL_LLAMA70B:-$ROOT_DIR/../models/Llama-3.3-70B-Instruct-AWQ}"
DATA_DIR="$ROOT_DIR/../15_fulldata_resplit/data"
OUT_DIR="$ROOT_DIR/data"

mkdir -p "$OUT_DIR" "$ROOT_DIR/logs"

echo "=== Generating Llama-70B teacher labels (train) with real logprobs ==="
"$PY" "$SHARED_DIR/generate_teacher_labels_local_logprobs.py" \
  --model_path "$MODEL" \
  --dataset "$DATA_DIR/train.jsonl" \
  --output "$OUT_DIR/teacher_train.jsonl" \
  --gt_field Answer \
  --resume \
  2>&1 | tee "$ROOT_DIR/logs/generate_teacher_train.log"

echo ""
echo "=== Generating Llama-70B teacher labels (test) with real logprobs ==="
"$PY" "$SHARED_DIR/generate_teacher_labels_local_logprobs.py" \
  --model_path "$MODEL" \
  --dataset "$DATA_DIR/test.jsonl" \
  --output "$OUT_DIR/teacher_test.jsonl" \
  --gt_field Answer \
  --resume \
  2>&1 | tee "$ROOT_DIR/logs/generate_teacher_test.log"

echo ""
echo "=== Done: teacher labels saved to $OUT_DIR ==="
