#!/usr/bin/env bash
# 一键评估封装（自包含，无外部相对依赖）
# 用法: bash scripts/run_eval.sh <BASE_MODEL_DIR> <TEST_JSONL>
set -euo pipefail

BASE_MODEL="${1:-}"
TEST_DATA="${2:-}"

if [[ -z "$BASE_MODEL" || -z "$TEST_DATA" ]]; then
  echo "用法: bash scripts/run_eval.sh <14B基座绝对路径> <测试集jsonl>"
  echo "示例: bash scripts/run_eval.sh /data/models/Qwen2.5-14B-Instruct data/test_full_991.jsonl"
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"

"$PY" "$ROOT_DIR/scripts/evaluate_model.py" \
  --base_model "$BASE_MODEL" \
  --adapter_dir "$ROOT_DIR/adapter" \
  --test_data "$TEST_DATA" \
  --wrong_log "$ROOT_DIR/test_wrong.jsonl"
