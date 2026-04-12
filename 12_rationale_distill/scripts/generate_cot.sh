#!/usr/bin/env bash
# 生成 Doubao CoT 推理链
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODULE_DIR="$(dirname "$SCRIPT_DIR")"
ROOT_DIR="$(dirname "$MODULE_DIR")"

source "${ROOT_DIR}/setup.env" 2>/dev/null || true

PYTHON="${EASYEDIT_PY:-python3}"

"${PYTHON}" "${ROOT_DIR}/shared/generate_teacher_cot.py" \
    --dataset "${MODULE_DIR}/data/train.jsonl" \
    --candidate "${MODULE_DIR}/configs/teacher_candidate.json" \
    --output "${MODULE_DIR}/data/train_cot.jsonl" \
    --gt_field Answer \
    --max_tokens 1024 \
    --filter_correct \
    --request_interval_sec 0.5 \
    --resume \
    2>&1 | tee "${MODULE_DIR}/logs/generate_cot.log"

echo ""
echo "=== CoT 生成完成 ==="
wc -l "${MODULE_DIR}/data/train_cot.jsonl"
