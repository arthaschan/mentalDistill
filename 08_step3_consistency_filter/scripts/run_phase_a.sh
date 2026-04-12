#!/usr/bin/env bash
# =============================================================================
# Stage 3 Phase A: DeepSeek 单教师真实软标签训练
# 先用已完成的 DeepSeek MV, 同时等 Doubao MV
# =============================================================================
set -euo pipefail

PROJ_DIR="/home/student/arthas/mentalDistill"
PYTHON="/home/student/anaconda3/bin/python3"
STEP3_DIR="$PROJ_DIR/08_step3_consistency_filter"
LOG_DIR="$STEP3_DIR/logs"
SHARED="$PROJ_DIR/shared"

BASE_MODEL="${BASE_MODEL_7B:-/home/student/arthas/EasyEdit3/Qwen2.5-7B-Instruct}"
VAL_DATA="$PROJ_DIR/00_baseline_gt_sft/data/val.jsonl"
TEST_DATA="$PROJ_DIR/00_baseline_gt_sft/data/test.jsonl"

mkdir -p "$LOG_DIR"

echo "=============================================="
echo "Stage 3 Phase A: DeepSeek Single Teacher MV"
echo "Start: $(date)"
echo "=============================================="

# ---- 数据已在上一步融合完成 ----
DS_FUSED="$STEP3_DIR/data/train_fused_ds_only.jsonl"
if [[ ! -s "$DS_FUSED" ]]; then
    echo "[ERROR] Fused data not found: $DS_FUSED"
    exit 1
fi
echo "[DATA] $(wc -l < "$DS_FUSED") samples"

# CSV
echo "label,alpha,lr,epochs,seed,val_best,test_acc" > "$STEP3_DIR/phase_a_results.csv"

run_train() {
    local LABEL="$1" ALPHA="$2" LR="$3" EPOCHS="$4" SEED="$5" DATA="$6"
    local OUT_DIR="$STEP3_DIR/outputs/$LABEL"

    echo ""
    echo "=== [TRAIN] $LABEL  α=$ALPHA lr=$LR ep=$EPOCHS seed=$SEED ==="

    $PYTHON "$SHARED/train_choice_head_distill.py" \
        --model_name "$BASE_MODEL" \
        --data_path "$DATA" \
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
        2>&1 | tee "$LOG_DIR/train_${LABEL}.log"

    local BEST_VAL=$(grep "\[BEST\]" "$LOG_DIR/train_${LABEL}.log" | tail -1 | grep -oP "val_acc=\K[0-9.]+" || echo "?")
    local TEST_ACC=$(grep "\[TEST-BEST\]" "$LOG_DIR/train_${LABEL}.log" | tail -1 | grep -oP "test_acc=\K[0-9.]+" || echo "?")
    echo "[RESULT] $LABEL → val=$BEST_VAL% test=$TEST_ACC%"
    echo "$LABEL,$ALPHA,$LR,$EPOCHS,$SEED,$BEST_VAL,$TEST_ACC" >> "$STEP3_DIR/phase_a_results.csv"
}

# Phase A configs: DeepSeek-only real soft labels
# A1: α=0.9 (strong KD weight) - baseline
run_train "ds_real_a09_s42" 0.9 1e-4 6 42 "$DS_FUSED"

# A2: α=0.5 (balanced) 
run_train "ds_real_a05_s42" 0.5 1e-4 6 42 "$DS_FUSED"

# A3: α=0.1 (like Round3 best)
run_train "ds_real_a01_s42" 0.1 1e-4 6 42 "$DS_FUSED"

# A4: α=0.1 seed=8 (Round3 best seed)
run_train "ds_real_a01_s8" 0.1 1e-4 6 8 "$DS_FUSED"

echo ""
echo "=============================================="
echo "Phase A Done: $(date)"
echo "=============================================="
cat "$STEP3_DIR/phase_a_results.csv" | column -t -s ','
