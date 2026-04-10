#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/../shared/common_env.sh"
resolve_python
resolve_model_dir BASE_MODEL_7B Qwen2.5-7B-Instruct

LATEST_RUN="$(ls -td "$ROOT_DIR"/runs/* 2>/dev/null | head -1)"
if [[ -z "$LATEST_RUN" ]]; then
  echo "未找到 runs 目录，请先训练"
  exit 1
fi

ADAPTER_DIR="$(find "$LATEST_RUN/outputs" -type d -path '*/stage2_sft/best' | head -1)"
if [[ -z "$ADAPTER_DIR" ]]; then
  echo "未找到 stage2_sft/best，请检查训练结果"
  exit 1
fi

"$PY" "$SHARED_DIR/evaluate_model.py" \
  --base_model "$BASE_MODEL_7B" \
  --adapter_dir "$ADAPTER_DIR" \
  --test_data "$ROOT_DIR/data/test.jsonl" \
  --wrong_log "$ROOT_DIR/outputs/test_wrong.jsonl"
