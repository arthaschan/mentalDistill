#!/usr/bin/env bash
# Round 3: 两阶段训练 + 单阶段种子/微调搜索
# 目标: 超过单老师最佳 81.93%
#
# Group A: 两阶段训练 (KL distill → GT SFT)，复刻单老师 81.93% 的成功经验
# Group B: 单阶段 α=0.9 lr=1e-4 的种子扫描 + rank=32 变体
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"
SHARED="$REPO_ROOT/shared"

source "$REPO_ROOT/setup.env"
PY="${EASYEDIT_PY:-python3}"

echo "Round 3: 两阶段+种子扫描 - $(date)"

TRAIN_HEAD="$PROJECT_ROOT/data/train_fused_4t_eps0.05.jsonl"
TRAIN_GT="$REPO_ROOT/00_baseline_gt_sft/data/train.jsonl"
VAL_DATA="$REPO_ROOT/00_baseline_gt_sft/data/val.jsonl"
TEST_DATA="$REPO_ROOT/00_baseline_gt_sft/data/test.jsonl"
BASE_MODEL="${BASE_MODEL_7B}"

LOG_DIR="$REPO_ROOT/night_logs"
mkdir -p "$LOG_DIR"
LOGFILE="$LOG_DIR/round3_two_stage.log"

# ============================================================
# Group A: 两阶段训练 (7 配置)
# ============================================================
echo "====== Group A: Two-Stage Training ======" | tee -a "$LOGFILE"

$PY "$SHARED/run_two_stage_training.py" \
    --params "$PROJECT_ROOT/configs/round3_two_stage_params.json" \
    --run_root "$PROJECT_ROOT" \
    --project_root "$REPO_ROOT" \
    --base_model "$BASE_MODEL" \
    --train_head "$TRAIN_HEAD" \
    --train_gt "$TRAIN_GT" \
    --val_data "$VAL_DATA" \
    --test_data "$TEST_DATA" \
    --teacher_prefix "fused4t" \
    --py "$PY" 2>&1 | tee -a "$LOGFILE"

echo "" | tee -a "$LOGFILE"
echo "====== Group A Done ======" | tee -a "$LOGFILE"

# ============================================================
# Group B: 单阶段种子扫描 + rank=32 (6 配置)
# ============================================================
echo "====== Group B: Single-Stage Seed + Rank Sweep ======" | tee -a "$LOGFILE"

CONFIGS_B=(
    # 种子扫描: α=0.9, lr=1e-4, rank=16
    "r3b_seed2    0.1 1e-4 16 32 8 2"
    "r3b_seed11   0.1 1e-4 16 32 11 2"
    "r3b_seed7    0.1 1e-4 16 32 7 2"
    # rank=32 变体
    "r3b_r32_s42  0.1 1e-4 32 64 42 2"
    "r3b_r32_s11  0.1 1e-4 32 64 11 2"
    # α=0.85, lr=1e-4, seed=42
    "r3b_a85_s42  0.15 1e-4 16 32 42 2"
)

RUN_IDX=0
for cfg in "${CONFIGS_B[@]}"; do
    read -r NAME CODE_ALPHA LR RANK LALPHA SEED BSACC <<< "$cfg"
    RUN_IDX=$((RUN_IDX + 1))
    OUTDIR="$PROJECT_ROOT/outputs/r3_${NAME}"
    echo "=== Group B Run $RUN_IDX: $NAME | code_α=$CODE_ALPHA lr=$LR rank=$RANK seed=$SEED ===" | tee -a "$LOGFILE"

    $PY "$SHARED/train_choice_head_distill.py" \
        --model_name "$BASE_MODEL" \
        --data_path "$TRAIN_HEAD" \
        --val_path "$VAL_DATA" \
        --test_path "$TEST_DATA" \
        --output_dir "$OUTDIR" \
        --num_epochs 8 \
        --batch_size "$BSACC" \
        --gradient_accumulation_steps 8 \
        --learning_rate "$LR" \
        --rank "$RANK" \
        --lora_alpha "$LALPHA" \
        --alpha "$CODE_ALPHA" \
        --seed "$SEED" \
        --warmup_ratio 0.1 \
        --use_cosine_schedule \
        --deterministic 2>&1 | tee -a "$LOGFILE"

    echo "" | tee -a "$LOGFILE"
done

echo "====== Group B Done ======" | tee -a "$LOGFILE"

# ============================================================
# Summary
# ============================================================
echo "" | tee -a "$LOGFILE"
echo "====== Round 3 Summary ======" | tee -a "$LOGFILE"

echo "--- Group A (Two-Stage) Results ---" | tee -a "$LOGFILE"
for lf in "$PROJECT_ROOT"/logs/stage2_fused4t_*.log; do
    if [[ -f "$lf" ]]; then
        NAME=$(basename "$lf" .log | sed 's/stage2_//')
        BEST_VAL=$(grep '验证准确率' "$lf" | awk -F': ' '{print $2}' | sort -t'%' -k1 -nr | head -1)
        TEST_ACC=$(grep '测试集准确率' "$lf" | awk -F': ' '{print $2}')
        echo "  $NAME → best_val=$BEST_VAL test=$TEST_ACC" | tee -a "$LOGFILE"
    fi
done

echo "--- Group B (Single-Stage) Results ---" | tee -a "$LOGFILE"
grep -E '(=== Group B Run|BEST|TEST-BEST)' "$LOGFILE"

echo ""
echo "Round 3 finished: $(date)" | tee -a "$LOGFILE"
