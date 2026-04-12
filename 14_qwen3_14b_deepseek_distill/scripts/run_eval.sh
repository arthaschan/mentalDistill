#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/../shared/common_env.sh"
resolve_python
resolve_model_dir BASE_MODEL_QWEN3_8B Qwen3-8B

LATEST_RUN="$(ls -td "$ROOT_DIR"/runs/* 2>/dev/null | head -1)"
if [[ -z "$LATEST_RUN" ]]; then
  echo "未找到 runs 目录，请先训练"
  exit 1
fi

ADAPTER_DIR="$(find "$LATEST_RUN/outputs" -type d -path '*/stage2_sft/best' | head -1)"
if [[ -z "$ADAPTER_DIR" ]]; then
  # 回退到 stage1 best（Module 13 经验：Stage1 可能优于 Stage2）
  ADAPTER_DIR="$(find "$LATEST_RUN/outputs" -type d -path '*/stage1_head/best' | head -1)"
fi
if [[ -z "$ADAPTER_DIR" ]]; then
  echo "未找到 adapter 目录，请检查训练结果"
  exit 1
fi

echo "评估 adapter: $ADAPTER_DIR"

"$PY" "$SHARED_DIR/evaluate_model.py" \
  --base_model "$BASE_MODEL_QWEN3_8B" \
  --adapter_dir "$ADAPTER_DIR" \
  --test_data "$ROOT_DIR/data/test.jsonl" \
  --wrong_log "$ROOT_DIR/outputs/test_wrong.jsonl"
