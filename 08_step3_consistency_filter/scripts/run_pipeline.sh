#!/usr/bin/env bash
# =============================================================================
# Stage 3: 真实多票软标签融合 + 自一致性过滤 → 训练 sweep
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

DOUBAO_MV="$PROJ_DIR/03_doubao_choice_head/data/teacher_train_multivote.jsonl"
DEEPSEEK_MV="$PROJ_DIR/02_deepseek_v3_choice_head/data/teacher_train_multivote.jsonl"

# 加载环境
[[ -f "$PROJ_DIR/setup.env" ]] && source "$PROJ_DIR/setup.env"

mkdir -p "$LOG_DIR" "$STEP3_DIR/data"

echo "=============================================="
echo "Stage 3: 真实多票融合 + 自一致性过滤 Sweep"
echo "Start: $(date)"
echo "=============================================="

# ---- 检查多票数据是否就绪 ----
for f in "$DOUBAO_MV" "$DEEPSEEK_MV"; do
    if [[ ! -s "$f" ]]; then
        echo "[ERROR] 多票数据不存在或为空: $f"
        echo "请先运行 multi-vote 采样脚本"
        exit 1
    fi
done

echo "[CHECK] Doubao multivote: $(wc -l < "$DOUBAO_MV") samples"
echo "[CHECK] DeepSeek multivote: $(wc -l < "$DEEPSEEK_MV") samples"

# ---- Sweep 配置 ----
# 从 JSON 读取 sweep_configs
CONFIGS_JSON="$STEP3_DIR/configs/stage3_params.json"

run_single_stage() {
    local LABEL="$1" ALPHA="$2" LR="$3" EPOCHS="$4" SEED="$5"
    local HT="${6:-0.78}" LT="${7:-0.56}" DWF="${8:-0.5}"
    local DATA_FILE="$STEP3_DIR/data/train_fused_${LABEL}.jsonl"
    local OUT_DIR="$STEP3_DIR/outputs/$LABEL"
    local REPORT="$STEP3_DIR/data/report_${LABEL}.json"

    echo ""
    echo "================================================"
    echo "[RUN] $LABEL  α=$ALPHA lr=$LR ep=$EPOCHS seed=$SEED"
    echo "  filter: high=$HT low=$LT downweight=$DWF"
    echo "================================================"

    # Step 1: 融合
    $PYTHON "$SHARED/fuse_multivote_teachers.py" \
        --base "$BASE_DATA" \
        --teachers "$DOUBAO_MV" "$DEEPSEEK_MV" \
        --weights 0.528 0.472 \
        --teacher_names Doubao DeepSeek \
        --output "$DATA_FILE" \
        --high_thresh "$HT" \
        --low_thresh "$LT" \
        --downweight_factor "$DWF" \
        --report "$REPORT" \
        2>&1 | tee "$LOG_DIR/fuse_${LABEL}.log"

    # Step 2: 训练
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
        2>&1 | tee "$LOG_DIR/train_${LABEL}.log"

    # 提取结果
    local BEST_VAL=$(grep "\[BEST\]" "$LOG_DIR/train_${LABEL}.log" | tail -1 | grep -oP "val_acc=\K[0-9.]+")
    local TEST_ACC=$(grep "\[TEST-BEST\]" "$LOG_DIR/train_${LABEL}.log" | tail -1 | grep -oP "test_acc=\K[0-9.]+")
    echo "[RESULT] $LABEL → val_best=$BEST_VAL test=$TEST_ACC"
    echo "$LABEL,$ALPHA,$LR,$EPOCHS,$SEED,$HT,$LT,$DWF,$BEST_VAL,$TEST_ACC" >> "$STEP3_DIR/sweep_results.csv"
}

run_two_stage() {
    local LABEL="$1" ALPHA="$2" LR="$3" S1_EP="$4" S2_EP="$5" SEED="$6"
    local HT="${7:-0.78}" LT="${8:-0.56}" DWF="${9:-0.5}"
    local DATA_FILE="$STEP3_DIR/data/train_fused_${LABEL}.jsonl"
    local OUT_DIR="$STEP3_DIR/outputs/$LABEL"
    local REPORT="$STEP3_DIR/data/report_${LABEL}.json"

    echo ""
    echo "================================================"
    echo "[TWO-STAGE] $LABEL  α=$ALPHA lr=$LR s1=$S1_EP+s2=$S2_EP seed=$SEED"
    echo "================================================"

    # Step 1: 融合
    $PYTHON "$SHARED/fuse_multivote_teachers.py" \
        --base "$BASE_DATA" \
        --teachers "$DOUBAO_MV" "$DEEPSEEK_MV" \
        --weights 0.528 0.472 \
        --teacher_names Doubao DeepSeek \
        --output "$DATA_FILE" \
        --high_thresh "$HT" \
        --low_thresh "$LT" \
        --downweight_factor "$DWF" \
        --report "$REPORT" \
        2>&1 | tee "$LOG_DIR/fuse_${LABEL}.log"

    # Step 2: 生成参数文件
    local PARAMS_FILE="$STEP3_DIR/data/params_${LABEL}.json"
    cat > "$PARAMS_FILE" <<PARAMEOF
[{
    "name": "${LABEL}",
    "alpha_stage1": $ALPHA,
    "learning_rate_stage1": $LR,
    "learning_rate_stage2": $LR,
    "num_epochs_stage1": $S1_EP,
    "num_epochs_stage2": $S2_EP,
    "seed": $SEED,
    "rank": 16,
    "lora_alpha": 32,
    "batch_size": 2,
    "gradient_accumulation_steps": 8
}]
PARAMEOF

    # Step 3: 两阶段训练
    $PYTHON "$SHARED/run_two_stage_training.py" \
        --base_model "$BASE_MODEL" \
        --train_head "$DATA_FILE" \
        --train_gt "$BASE_DATA" \
        --val_data "$VAL_DATA" \
        --test_data "$TEST_DATA" \
        --run_root "$OUT_DIR" \
        --project_root "$PROJ_DIR" \
        --teacher_prefix "mt" \
        --params "$PARAMS_FILE" \
        --py "$PYTHON" \
        2>&1 | tee "$LOG_DIR/train_${LABEL}.log"

    # 提取结果 (两阶段的结果在 stage2 日志中)
    local TEST_ACC=$(grep "测试集准确率" "$LOG_DIR/train_${LABEL}.log" | tail -1 | grep -oP "[0-9]+\.[0-9]+")
    echo "[RESULT] $LABEL → test=$TEST_ACC"
    echo "$LABEL,$ALPHA,$LR,${S1_EP}+${S2_EP},$SEED,$HT,$LT,$DWF,,$TEST_ACC" >> "$STEP3_DIR/sweep_results.csv"
}

# ---- CSV header ----
echo "label,alpha,lr,epochs,seed,high_thresh,low_thresh,downweight,val_best,test_acc" > "$STEP3_DIR/sweep_results.csv"

# ---- 执行 sweep ----
# Config 1: baseline 2-teacher real labels
run_single_stage "2t_real_a09" 0.9 1e-4 6 42

# Config 2: lower GT weight
run_single_stage "2t_real_a07" 0.7 1e-4 6 42

# Config 3: balanced
run_single_stage "2t_real_a05" 0.5 1e-4 6 42

# Config 4-5: seed robustness
run_single_stage "2t_real_a09_s11" 0.9 1e-4 6 11
run_single_stage "2t_real_a09_s8"  0.9 1e-4 6 8

# Config 6: no filter (ablation)
run_single_stage "2t_nofilter" 0.9 1e-4 6 42 0.0 0.0 0.5

# Config 7: strict filter
run_single_stage "2t_strict" 0.9 1e-4 6 42 0.89 0.67 0.5

# Config 8: two-stage (like DeepSeek best)
run_two_stage "2t_twostage" 0.35 1.2e-4 1 2 42

echo ""
echo "=============================================="
echo "Stage 3 Sweep DONE: $(date)"
echo "=============================================="
echo ""
echo "Results:"
cat "$STEP3_DIR/sweep_results.csv" | column -t -s ','
