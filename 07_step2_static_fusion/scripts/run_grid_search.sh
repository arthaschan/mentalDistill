#!/usr/bin/env bash
# =============================================================================
# 阶段2: 蒸馏训练 — α/T 网格搜索 (粗搜 10 组)
#
# 注意: 训练代码中 alpha = KL权重 (蒸馏), 1-alpha = CE权重 (GT)
# 方案文档中 α = GT权重, β = KL权重 = 1-α
# 映射: 文档α=0.3 → 代码alpha=0.7
# =============================================================================
set -euo pipefail

PROJ_DIR="/home/student/arthas/mentalDistill"
PYTHON="/home/student/anaconda3/bin/python3"
STEP2_DIR="$PROJ_DIR/07_step2_static_fusion"
LOG_FILE="$PROJ_DIR/night_logs/stage2_grid_search.log"

mkdir -p "$STEP2_DIR/data" "$STEP2_DIR/configs" "$PROJ_DIR/night_logs"

# 加载环境
if [[ -f "$PROJ_DIR/setup.env" ]]; then
    source "$PROJ_DIR/setup.env"
fi

exec > >(tee -a "$LOG_FILE") 2>&1

MODEL="${BASE_MODEL_7B:-/home/student/arthas/EasyEdit3/Qwen2.5-7B-Instruct}"
FUSED_DATA="$STEP2_DIR/data/train_fused_4t.jsonl"
VAL_DATA="$PROJ_DIR/00_baseline_gt_sft/data/val.jsonl"
TEST_DATA="$PROJ_DIR/00_baseline_gt_sft/data/test.jsonl"

echo "========================================"
echo "阶段2: 蒸馏训练网格搜索 - $(date)"
echo "========================================"
echo "Model: $MODEL"
echo "Fused data: $FUSED_DATA ($(wc -l < "$FUSED_DATA") lines)"
echo "Val: $VAL_DATA, Test: $TEST_DATA"
echo ""

# ---- Step 1: 预处理不同温度的标签 ----
echo "Step 1: 生成不同温度的标签..."
for T in 1.0 1.5 2.0 3.0; do
    OUT="$STEP2_DIR/data/train_fused_4t_T${T}.jsonl"
    if [[ "$T" == "1.0" ]]; then
        cp "$FUSED_DATA" "$OUT"
        echo "  T=$T: copied (no change)"
    else
        $PYTHON "$PROJ_DIR/shared/apply_temperature.py" \
            --input "$FUSED_DATA" \
            --output "$OUT" \
            --temperature "$T"
    fi
done
echo ""

# ---- Step 2: 网格搜索 ----
# 文档α (GT权重) → 代码alpha (KL权重): code_alpha = 1 - doc_alpha
# 文档 α=0.1,0.3,0.5 → 代码 alpha=0.9,0.7,0.5

declare -A GRID_RESULTS
BEST_ACC=0
BEST_KEY=""

echo "========================================"
echo "Step 2: 10组网格搜索"
echo "========================================"
echo ""
echo "| # | doc_α (GT) | T | code_alpha (KL) | Val Acc | Test Acc |"
echo "|---|-----------|---|----------------|---------|----------|"

RUN_ID=0
for DOC_ALPHA in 0.1 0.3 0.5; do
    CODE_ALPHA=$(python3 -c "print(round(1.0 - $DOC_ALPHA, 2))")
    
    for T in 1.0 1.5 2.0 3.0; do
        # 跳过不在计划中的组合 (doc_α=0.1只搜T=1,2,3; doc_α=0.3搜全部; doc_α=0.5搜T=1,2,3)
        if [[ "$DOC_ALPHA" == "0.1" && "$T" == "1.5" ]]; then continue; fi
        if [[ "$DOC_ALPHA" == "0.5" && "$T" == "1.5" ]]; then continue; fi
        
        RUN_ID=$((RUN_ID + 1))
        RUN_KEY="a${DOC_ALPHA}_T${T}"
        DATA_FILE="$STEP2_DIR/data/train_fused_4t_T${T}.jsonl"
        OUT_DIR="$STEP2_DIR/outputs/grid_${RUN_KEY}"

        echo ""
        echo "=== Run $RUN_ID: doc_α=$DOC_ALPHA, T=$T, code_alpha=$CODE_ALPHA ==="
        
        $PYTHON "$PROJ_DIR/shared/train_choice_head_distill.py" \
            --model_name "$MODEL" \
            --data_path "$DATA_FILE" \
            --val_path "$VAL_DATA" \
            --test_path "$TEST_DATA" \
            --output_dir "$OUT_DIR" \
            --num_epochs 4 \
            --batch_size 2 \
            --gradient_accumulation_steps 8 \
            --learning_rate 1e-4 \
            --rank 16 \
            --lora_alpha 32 \
            --alpha "$CODE_ALPHA" \
            --default_distill_mask 1 \
            --seed 42 \
            --deterministic 2>&1 | tail -20

        # 提取测试集准确率
        TEST_ACC=$(grep "测试集准确率" "$OUT_DIR"/../grid_${RUN_KEY}_stdout.log 2>/dev/null | grep -oP '\d+\.\d+' || echo "N/A")
        
        # 如果没有从stdout拿到，尝试从最后输出获取
        if [[ "$TEST_ACC" == "N/A" ]]; then
            # 从 tee 日志获取
            TEST_ACC=$(grep -A0 "测试集准确率" "$LOG_FILE" | tail -1 | grep -oP '\d+\.\d+' || echo "N/A")
        fi
        
        echo "| $RUN_ID | $DOC_ALPHA | $T | $CODE_ALPHA | see_log | see_log |"
        
        # 保存配置
        cat > "$OUT_DIR/run_config.json" <<CONF
{
  "doc_alpha_gt_weight": $DOC_ALPHA,
  "code_alpha_kl_weight": $CODE_ALPHA,
  "temperature": $T,
  "run_id": $RUN_ID,
  "run_key": "$RUN_KEY",
  "model": "$MODEL",
  "data": "$DATA_FILE",
  "epochs": 4,
  "batch_size": 2,
  "grad_acc": 8,
  "lr": 1e-4,
  "rank": 16,
  "lora_alpha": 32
}
CONF
    done
done

echo ""
echo "========================================"
echo "Step 3: 汇总结果"
echo "========================================"

# 从日志中提取所有 val/test 准确率
echo ""
echo "提取各 run 的准确率..."
echo ""
echo "| Run | doc_α | T | Val Acc (best epoch) | Test Acc |"
echo "|-----|-------|---|---------------------|----------|"

for d in "$STEP2_DIR"/outputs/grid_*; do
    if [[ ! -d "$d" ]]; then continue; fi
    RUN_KEY=$(basename "$d" | sed 's/grid_//')
    
    # 从 run_config 获取参数
    DA=$(python3 -c "import json; print(json.load(open('$d/run_config.json'))['doc_alpha_gt_weight'])" 2>/dev/null || echo "?")
    TT=$(python3 -c "import json; print(json.load(open('$d/run_config.json'))['temperature'])" 2>/dev/null || echo "?")
    
    # 查找 val acc (最佳 epoch)
    BEST_VAL="N/A"
    
    # 查找 test acc
    TTEST="N/A"
    
    echo "| $RUN_KEY | $DA | $TT | $BEST_VAL | $TTEST |"
done

echo ""
echo "========================================"
echo "阶段2 网格搜索完成 - $(date)"
echo "========================================"
echo ""
echo "查看详细日志: $LOG_FILE"
echo "各 run 输出: $STEP2_DIR/outputs/grid_*/"
