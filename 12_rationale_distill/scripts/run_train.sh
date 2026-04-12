#!/usr/bin/env bash
# 推理链蒸馏 SFT 训练
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODULE_DIR="$(dirname "$SCRIPT_DIR")"
ROOT_DIR="$(dirname "$MODULE_DIR")"

source "${ROOT_DIR}/setup.env" 2>/dev/null || true
source "${ROOT_DIR}/shared/common_env.sh" 2>/dev/null || true

PYTHON="${EASYEDIT_PY:-python3}"
BASE_MODEL="${BASE_MODEL_7B:-${MODELS_DIR:-/home/student/arthas/EasyEdit3}/Qwen2.5-7B-Instruct}"
SEED="${SEED:-42}"

TRAIN_DATA="${MODULE_DIR}/data/train_cot.jsonl"
VAL_DATA="${MODULE_DIR}/data/val.jsonl"
OUTPUT_DIR="${MODULE_DIR}/outputs/seed_${SEED}"

if [[ ! -f "$TRAIN_DATA" ]]; then
    echo "ERROR: train_cot.jsonl 不存在，请先运行 generate_cot.sh"
    exit 1
fi

echo "=== 推理链蒸馏 SFT 训练 ==="
echo "  Seed: ${SEED}"
echo "  Train: $(wc -l < "$TRAIN_DATA") 条 CoT 样本"
echo "  Model: ${BASE_MODEL}"

"${PYTHON}" "${ROOT_DIR}/shared/train_rationale_sft.py" \
    --base_model "${BASE_MODEL}" \
    --train_data "${TRAIN_DATA}" \
    --val_data "${VAL_DATA}" \
    --output_dir "${OUTPUT_DIR}" \
    --epochs 6 \
    --lr 1e-4 \
    --batch_size 2 \
    --max_length 2048 \
    --seed "${SEED}" \
    --lora_rank 16 \
    --lora_alpha 32 \
    --warmup_ratio 0.1 \
    --deterministic \
    2>&1 | tee "${MODULE_DIR}/logs/train_seed_${SEED}.log"
