#!/usr/bin/env bash
# 推理链蒸馏模型评估
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODULE_DIR="$(dirname "$SCRIPT_DIR")"
ROOT_DIR="$(dirname "$MODULE_DIR")"

source "${ROOT_DIR}/setup.env" 2>/dev/null || true
source "${ROOT_DIR}/shared/common_env.sh" 2>/dev/null || true

PYTHON="${EASYEDIT_PY:-python3}"
BASE_MODEL="${BASE_MODEL_7B:-${MODELS_DIR:-/home/student/arthas/EasyEdit3}/Qwen2.5-7B-Instruct}"
SEED="${SEED:-42}"

ADAPTER_DIR="${MODULE_DIR}/outputs/seed_${SEED}/best"
TEST_DATA="${MODULE_DIR}/data/test.jsonl"
WRONG_LOG="${MODULE_DIR}/outputs/seed_${SEED}/test_wrong.jsonl"

echo "=== 推理链蒸馏模型评估 ==="
echo "  Adapter: ${ADAPTER_DIR}"

"${PYTHON}" "${ROOT_DIR}/shared/evaluate_model.py" \
    --base_model "${BASE_MODEL}" \
    --adapter_dir "${ADAPTER_DIR}" \
    --test_data "${TEST_DATA}" \
    --wrong_log "${WRONG_LOG}"
