#!/usr/bin/env bash
# =============================================================================
# Stage 3 快速验证: DeepSeek 单教师真实多票标签
# (不等 Doubao 完成，先用 DeepSeek 独立测试真实分布的效果)
# =============================================================================
set -euo pipefail

PROJ_DIR="/home/student/arthas/mentalDistill"
PYTHON="/home/student/anaconda3/bin/python3"
STEP3_DIR="$PROJ_DIR/08_step3_consistency_filter"
LOG_DIR="$STEP3_DIR/logs"
SHARED="$PROJ_DIR/shared"

BASE_MODEL="${BASE_MODEL_7B:-/home/student/arthas/EasyEdit3/Qwen2.5-7B-Instruct}"
BASE_DATA="$PROJ_DIR/00_baseline_gt_sft/data/train.jsonl"
VAL_DATA="$PROJ_DIR/00_baseline_gt_sft/data/val.jsonl"
TEST_DATA="$PROJ_DIR/00_baseline_gt_sft/data/test.jsonl"

DEEPSEEK_MV="$PROJ_DIR/02_deepseek_v3_choice_head/data/teacher_train_multivote.jsonl"

[[ -f "$PROJ_DIR/setup.env" ]] && source "$PROJ_DIR/setup.env"

mkdir -p "$LOG_DIR" "$STEP3_DIR/data"

echo "=============================================="
echo "Quick test: DeepSeek 单教师真实多票标签"
echo "Start: $(date)"
echo "=============================================="

if [[ ! -s "$DEEPSEEK_MV" ]]; then
    echo "[ERROR] DeepSeek 多票数据不存在: $DEEPSEEK_MV"
    exit 1
fi
echo "[CHECK] DeepSeek multivote: $(wc -l < "$DEEPSEEK_MV") samples"

# 融合: 只用DeepSeek一个教师 (权重=1.0, 无需融合, 直接用作训练数据)
# 但通过 fuse_multivote_teachers 仍可以做自一致性过滤
run_config() {
    local LABEL="$1" ALPHA="$2" LR="$3" EPOCHS="$4" SEED="$5"
    local HT="${6:-0.78}" LT="${7:-0.56}" DWF="${8:-0.5}"
    local DATA_FILE="$STEP3_DIR/data/train_ds_real_${LABEL}.jsonl"
    local OUT_DIR="$STEP3_DIR/outputs/ds_${LABEL}"
    local REPORT="$STEP3_DIR/data/report_ds_${LABEL}.json"

    echo ""
    echo "================================================"
    echo "[RUN] ds_$LABEL  α=$ALPHA lr=$LR ep=$EPOCHS seed=$SEED"
    echo "================================================"

    # 融合 (单教师)
    $PYTHON "$SHARED/fuse_multivote_teachers.py" \
        --base "$BASE_DATA" \
        --teachers "$DEEPSEEK_MV" \
        --weights 1.0 \
        --teacher_names DeepSeek \
        --output "$DATA_FILE" \
        --high_thresh "$HT" \
        --low_thresh "$LT" \
        --downweight_factor "$DWF" \
        --report "$REPORT" \
        2>&1 | tee "$LOG_DIR/fuse_ds_${LABEL}.log"

    # 训练
    $PYTHON "$SHARED/train_choice_head_distill.py" \
        --model_name "$BASE_MODEL" \
        --data_path "$DATA_FILE" \
        --val_path "$VAL_DATA" \
        --test_path "$TEST_DATA" \
        --output_dir "$OUT_DIR" \
        --num_epochs "$EPOCHS" \
        --batch_size 2 \
        --gradient_accumulation_steps 8 \
        --learning_rate "$LR" \
        --rank 16 \
        --lora_alpha 32 \
        --alpha "$ALPHA" \
        --seed "$SEED" \
        --warmup_ratio 0.1 \
        --use_cosine_schedule \
        --deterministic \
        2>&1 | tee "$LOG_DIR/train_ds_${LABEL}.log"

    local BEST_VAL=$(grep "\[BEST\]" "$LOG_DIR/train_ds_${LABEL}.log" | tail -1 | grep -oP "val_acc=\K[0-9.]+")
    local TEST_ACC=$(grep "\[TEST-BEST\]" "$LOG_DIR/train_ds_${LABEL}.log" | tail -1 | grep -oP "test_acc=\K[0-9.]+")
    echo "[RESULT] ds_$LABEL → val_best=$BEST_VAL test=$TEST_ACC"
    echo "ds_$LABEL,$ALPHA,$LR,$EPOCHS,$SEED,$HT,$LT,$DWF,$BEST_VAL,$TEST_ACC" >> "$STEP3_DIR/ds_quick_results.csv"
}

echo "label,alpha,lr,epochs,seed,high_thresh,low_thresh,downweight,val_best,test_acc" > "$STEP3_DIR/ds_quick_results.csv"

# Config 1: 复刻 DeepSeek 最佳参数（α=0.35→对应 KL weight）+ 真实分布
run_config "a09" 0.9 1e-4 6 42

# Config 2: 更大 KL 权重
run_config "a07" 0.7 1e-4 6 42

# Config 3: α=0.5 平衡
run_config "a05" 0.5 1e-4 6 42

# Config 4: 不做一致性过滤 (ablation)
run_config "nofilter" 0.9 1e-4 6 42 0.0 0.0 0.5

echo ""
echo "=============================================="
echo "Quick test DONE: $(date)"
echo "=============================================="
cat "$STEP3_DIR/ds_quick_results.csv" | column -t -s ','
