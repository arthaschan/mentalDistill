#!/usr/bin/env bash
# =============================================================================
# 方案 A：多 Seed 集成训练 + Majority Vote 推理
# 最佳配置 (α=0.9, lr=1e-4, rank=16, 6 epochs) × 7 seeds
# =============================================================================
set -euo pipefail

PROJ_DIR="/home/student/arthas/mentalDistill"
PYTHON="/home/student/anaconda3/bin/python3"
SHARED="$PROJ_DIR/shared"
MODULE_DIR="$PROJ_DIR/11_multi_seed_ensemble"
LOG_DIR="$MODULE_DIR/logs"

BASE_MODEL="${BASE_MODEL_7B:-/home/student/arthas/EasyEdit3/Qwen2.5-7B-Instruct}"
FUSED_DATA="$PROJ_DIR/08_step3_consistency_filter/data/train_fused_2t_real_a09.jsonl"
VAL_DATA="$PROJ_DIR/00_baseline_gt_sft/data/val.jsonl"
TEST_DATA="$PROJ_DIR/00_baseline_gt_sft/data/test.jsonl"

[[ -f "$PROJ_DIR/setup.env" ]] && source "$PROJ_DIR/setup.env"

mkdir -p "$LOG_DIR" "$MODULE_DIR/outputs"

# 训练超参 (最佳配置)
ALPHA=0.9
LR=1e-4
EPOCHS=6
RANK=16
LORA_ALPHA=32

# 7 seeds
SEEDS=(7 8 11 42 55 77 123)

echo "=============================================="
echo "Plan A: Multi-Seed Ensemble Training"
echo "Seeds: ${SEEDS[*]}"
echo "Config: α=$ALPHA lr=$LR ep=$EPOCHS rank=$RANK"
echo "Start: $(date)"
echo "=============================================="

# 检查融合数据
if [[ ! -s "$FUSED_DATA" ]]; then
    echo "[ERROR] 融合训练数据不存在: $FUSED_DATA"
    echo "请确保 Step 3 的融合数据已生成"
    exit 1
fi
echo "[CHECK] 训练数据: $(wc -l < "$FUSED_DATA") samples"

# CSV header
RESULTS_CSV="$MODULE_DIR/seed_results.csv"
echo "seed,val_best,test_acc" > "$RESULTS_CSV"

# ---- 逐 seed 训练 ----
for SEED in "${SEEDS[@]}"; do
    OUT_DIR="$MODULE_DIR/outputs/seed_${SEED}"
    LOG_FILE="$LOG_DIR/train_seed_${SEED}.log"

    echo ""
    echo "================================================"
    echo "[TRAIN] seed=$SEED → $OUT_DIR"
    echo "================================================"

    $PYTHON "$SHARED/train_choice_head_distill.py" \
        --model_name "$BASE_MODEL" \
        --data_path "$FUSED_DATA" \
        --val_path "$VAL_DATA" \
        --test_path "$TEST_DATA" \
        --output_dir "$OUT_DIR" \
        --num_epochs "$EPOCHS" \
        --batch_size 2 \
        --gradient_accumulation_steps 8 \
        --learning_rate "$LR" \
        --rank "$RANK" \
        --lora_alpha "$LORA_ALPHA" \
        --alpha "$ALPHA" \
        --seed "$SEED" \
        --warmup_ratio 0.1 \
        --use_cosine_schedule \
        --deterministic \
        2>&1 | tee "$LOG_FILE"

    # 提取结果
    BEST_VAL=$(grep "\[BEST\]" "$LOG_FILE" | tail -1 | grep -oP "val_acc=\K[0-9.]+" || echo "N/A")
    TEST_ACC=$(grep "\[TEST-BEST\]" "$LOG_FILE" | tail -1 | grep -oP "test_acc=\K[0-9.]+" || echo "N/A")
    echo "[RESULT] seed=$SEED → val_best=$BEST_VAL test=$TEST_ACC"
    echo "$SEED,$BEST_VAL,$TEST_ACC" >> "$RESULTS_CSV"
done

echo ""
echo "=============================================="
echo "所有 seed 训练完成。Individual results:"
cat "$RESULTS_CSV" | column -t -s ','
echo "=============================================="

# ---- Majority Vote 集成推理 ----
echo ""
echo "=============================================="
echo "开始集成推理 (Majority Vote)..."
echo "=============================================="

# 收集所有 best adapter 目录
ADAPTER_DIRS=()
for SEED in "${SEEDS[@]}"; do
    BEST_DIR="$MODULE_DIR/outputs/seed_${SEED}/best"
    if [[ -d "$BEST_DIR" ]]; then
        ADAPTER_DIRS+=("$BEST_DIR")
    else
        echo "[WARN] seed=$SEED 的 best adapter 不存在，跳过"
    fi
done

if [[ ${#ADAPTER_DIRS[@]} -lt 3 ]]; then
    echo "[ERROR] 可用 adapter 不足 3 个，无法进行有效集成"
    exit 1
fi

$PYTHON "$SHARED/ensemble_majority_vote.py" \
    --base_model "$BASE_MODEL" \
    --adapter_dirs "${ADAPTER_DIRS[@]}" \
    --test_data "$TEST_DATA" \
    --output "$MODULE_DIR/ensemble_results.json" \
    2>&1 | tee "$LOG_DIR/ensemble_vote.log"

echo ""
echo "=============================================="
echo "Plan A DONE: $(date)"
echo "=============================================="
