#!/usr/bin/env bash
# 推理链蒸馏完整流水线：生成 CoT → 训练 → 评估
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODULE_DIR="$(dirname "$SCRIPT_DIR")"
ROOT_DIR="$(dirname "$MODULE_DIR")"

source "${ROOT_DIR}/setup.env" 2>/dev/null || true
source "${ROOT_DIR}/shared/common_env.sh" 2>/dev/null || true

PYTHON="${EASYEDIT_PY:-python3}"
BASE_MODEL="${BASE_MODEL_7B:-${MODELS_DIR:-/home/student/arthas/EasyEdit3}/Qwen2.5-7B-Instruct}"
SEED="${SEED:-42}"

echo "========================================"
echo "  12_rationale_distill 完整流水线"
echo "  Seed: ${SEED}"
echo "========================================"

# Step 1: 生成 CoT（如果尚不存在）
COT_FILE="${MODULE_DIR}/data/train_cot.jsonl"
if [[ ! -f "$COT_FILE" ]] || [[ $(wc -l < "$COT_FILE") -lt 100 ]]; then
    echo ""
    echo ">>> Step 1: 生成 Doubao CoT 推理链..."
    bash "${SCRIPT_DIR}/generate_cot.sh"
else
    echo ">>> Step 1: train_cot.jsonl 已存在 ($(wc -l < "$COT_FILE") 条)，跳过生成"
fi

# Step 2: 训练
echo ""
echo ">>> Step 2: 推理链 SFT 训练..."
bash "${SCRIPT_DIR}/run_train.sh"

# Step 3: 评估
echo ""
echo ">>> Step 3: 测试集评估..."
bash "${SCRIPT_DIR}/run_eval.sh"

echo ""
echo "========================================"
echo "  流水线完成"
echo "========================================"
