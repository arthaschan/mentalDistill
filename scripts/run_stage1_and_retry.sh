#!/usr/bin/env bash
# =============================================================================
# 阶段1: 标签融合 → 然后重跑失败的2个夜间定时任务
# =============================================================================
set -euo pipefail

PROJ_DIR="/home/student/arthas/mentalDistill"
PYTHON="/home/student/anaconda3/bin/python3"
LOG_DIR="$PROJ_DIR/night_logs"
LOG_FILE="$LOG_DIR/stage1_fusion_and_retry.log"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

# 加载环境
if [[ -f "$PROJ_DIR/setup.env" ]]; then
    source "$PROJ_DIR/setup.env"
fi

echo "========================================"
echo "阶段1: 标签融合 - $(date)"
echo "========================================"

# ---- 4老师融合 (主线，不含 Kimi) ----
echo ""
echo "[FUSION] 4老师静态加权融合 (Doubao/DeepSeek/Qwen-32B/Qwen-14B)"
echo "  权重: Doubao=0.285, DeepSeek=0.254, Qwen-32B=0.240, Qwen-14B=0.222"
echo ""

$PYTHON "$PROJ_DIR/shared/merge_teacher_soft_labels.py" \
    --base "$PROJ_DIR/00_baseline_gt_sft/data/train.jsonl" \
    --teachers \
        "$PROJ_DIR/03_doubao_choice_head/data/teacher_train_soft_full.jsonl" \
        "$PROJ_DIR/02_deepseek_v3_choice_head/data/teacher_train_soft.jsonl" \
        "$PROJ_DIR/01_qwen32b_whitebox/data/teacher_train_soft.jsonl" \
        "$PROJ_DIR/05_qwen14b_choice_head/data/teacher_train_soft.jsonl" \
    --weights 0.285 0.254 0.240 0.222 \
    --teacher_names Doubao DeepSeek Qwen-32B Qwen-14B \
    --output "$PROJ_DIR/07_step2_static_fusion/data/train_fused_4t.jsonl"

echo ""
echo "========================================"
echo "[FUSION] 5老师对照组 (含 Kimi)"
echo "========================================"

# 5老师权重: 按 accuracy 归一化 (98.80+87.95+83.13+77.11+61.45 = 408.44)
$PYTHON "$PROJ_DIR/shared/merge_teacher_soft_labels.py" \
    --base "$PROJ_DIR/00_baseline_gt_sft/data/train.jsonl" \
    --teachers \
        "$PROJ_DIR/03_doubao_choice_head/data/teacher_train_soft_full.jsonl" \
        "$PROJ_DIR/02_deepseek_v3_choice_head/data/teacher_train_soft.jsonl" \
        "$PROJ_DIR/01_qwen32b_whitebox/data/teacher_train_soft.jsonl" \
        "$PROJ_DIR/05_qwen14b_choice_head/data/teacher_train_soft.jsonl" \
        "$PROJ_DIR/04_kimi_choice_head/data/teacher_train_soft.jsonl" \
    --weights 0.242 0.215 0.204 0.189 0.150 \
    --teacher_names Doubao DeepSeek Qwen-32B Qwen-14B Kimi \
    --output "$PROJ_DIR/07_step2_static_fusion/data/train_fused_5t.jsonl"

echo ""
echo "========================================"
echo "阶段1 标签融合完成 - $(date)"
echo "========================================"

# ---- 重跑失败的2个夜间任务 ----
echo ""
echo "========================================"
echo "重跑夜间任务1: API Teacher Labels - $(date)"
echo "========================================"

export NIGHT_MODE=1
export REQUEST_INTERVAL_SEC="8"
export MAX_RETRIES="12"
export RATE_LIMIT_COOLDOWN_SEC="300"
export COOLDOWN_EVERY="30"
export COOLDOWN_SEC="120"

bash "$PROJ_DIR/02_deepseek_v3_choice_head/scripts/generate_teacher_labels.sh" || echo "[WARN] DeepSeek teacher labels failed"
bash "$PROJ_DIR/03_doubao_choice_head/scripts/generate_teacher_labels.sh" || echo "[WARN] Doubao teacher labels failed"

export REQUEST_INTERVAL_SEC="10"
export RATE_LIMIT_COOLDOWN_SEC="420"
export COOLDOWN_EVERY="15"
export COOLDOWN_SEC="180"
bash "$PROJ_DIR/03_doubao_choice_head/scripts/generate_teacher_soft_labels.sh" || echo "[WARN] Doubao soft labels failed"

export REQUEST_INTERVAL_SEC="8"
export RATE_LIMIT_COOLDOWN_SEC="300"
export COOLDOWN_EVERY="30"
export COOLDOWN_SEC="120"
bash "$PROJ_DIR/04_kimi_choice_head/scripts/generate_teacher_labels.sh" || echo "[WARN] Kimi teacher labels failed"

echo ""
echo "========================================"
echo "重跑夜间任务2: 批量训练 - $(date)"
echo "========================================"

export CONTINUE_ON_ERROR=1
bash "$PROJ_DIR/run_module_batch.sh" train

echo ""
echo "========================================"
echo "全部任务完成 - $(date)"
echo "========================================"
