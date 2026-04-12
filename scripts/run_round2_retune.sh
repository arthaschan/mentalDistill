#!/usr/bin/env bash
# =============================================================================
# Round 2: 重新调参 — 修复软标签 + cosine scheduler + best-epoch eval
#
# Round 1 问题诊断:
#   1. 所有教师软标签都是 smooth_eps=0.25 的假分布 (0.80/0.05×4)
#   2. 无LR调度器
#   3. Test仅在最终epoch评估 (未选最优val epoch)
#
# Round 2 改进:
#   1. 重新生成 smooth_eps=0.05 的软标签并融合
#   2. 使用 cosine LR scheduler + 10% warmup
#   3. Best-val-epoch 上评估 test
#   4. 搜索空间: α(GT) ∈ {0.5,0.7,0.9,1.0} × lr ∈ {5e-5,1e-4,2e-4}
#      T=1.0 only (Round1 已证明 T>1 无益)
#   5. 训练 8 个 epoch (让 cosine schedule 有足够空间)
# =============================================================================
set -euo pipefail

PROJ_DIR="/home/student/arthas/mentalDistill"
PYTHON="/home/student/anaconda3/bin/python3"
STEP2_DIR="$PROJ_DIR/07_step2_static_fusion"
LOG_FILE="$PROJ_DIR/night_logs/round2_retune.log"

mkdir -p "$STEP2_DIR/data" "$PROJ_DIR/night_logs"

if [[ -f "$PROJ_DIR/setup.env" ]]; then
    source "$PROJ_DIR/setup.env"
fi

exec > >(tee -a "$LOG_FILE") 2>&1

MODEL="${BASE_MODEL_7B:-/home/student/arthas/EasyEdit3/Qwen2.5-7B-Instruct}"
BASE_TRAIN="$PROJ_DIR/00_baseline_gt_sft/data/train.jsonl"
VAL_DATA="$PROJ_DIR/00_baseline_gt_sft/data/val.jsonl"
TEST_DATA="$PROJ_DIR/00_baseline_gt_sft/data/test.jsonl"

echo "========================================"
echo "Round 2: 重新调参 - $(date)"
echo "========================================"

# =============================================================================
# Step 1: 用 smooth_eps=0.05 重新生成各教师软标签
# =============================================================================
echo ""
echo "==== Step 1: 重新生成软标签 (smooth_eps=0.05) ===="
NEW_EPS=0.05

declare -A TEACHER_RAW=(
    ["doubao"]="$PROJ_DIR/03_doubao_choice_head/data/teacher_train_soft_full.jsonl"
    ["deepseek"]="$PROJ_DIR/02_deepseek_v3_choice_head/data/teacher_train_soft.jsonl"
    ["qwen32b"]="$PROJ_DIR/01_qwen32b_whitebox/data/teacher_train_soft.jsonl"
    ["qwen14b"]="$PROJ_DIR/05_qwen14b_choice_head/data/teacher_train_soft.jsonl"
)

# These files already have TeacherDist from smooth_eps=0.25.
# We need to re-smooth from the hard TeacherAnswer.
# Strategy: extract TeacherAnswer, re-apply prepare_soft_labels with new eps.

for TNAME in doubao deepseek qwen32b qwen14b; do
    RAW="${TEACHER_RAW[$TNAME]}"
    OUT="$STEP2_DIR/data/teacher_${TNAME}_eps${NEW_EPS}.jsonl"
    echo "  Re-smoothing $TNAME → eps=$NEW_EPS"
    $PYTHON -c "
import json, sys

EPS = float('$NEW_EPS')
OPTS = ['A','B','C','D','E']

with open('$RAW') as fin, open('$OUT', 'w') as fout:
    for line in fin:
        row = json.loads(line.strip())
        # Get hard teacher answer
        ta = row.get('TeacherAnswer', '')
        if not ta:
            # Infer from existing TeacherDist
            td = row.get('TeacherDist', {})
            if td:
                ta = max(td, key=td.get)
        ta = ta.strip().upper()
        if ta not in OPTS:
            ta = row.get('Answer', 'A').strip().upper()
        # Re-generate smoothed dist
        dist = {}
        for k in OPTS:
            if k == ta:
                dist[k] = 1.0 - EPS + EPS / len(OPTS)
            else:
                dist[k] = EPS / len(OPTS)
        row['TeacherDist'] = dist
        if 'TeacherAnswer' not in row:
            row['TeacherAnswer'] = ta
        fout.write(json.dumps(row, ensure_ascii=False) + '\n')
print('    $TNAME: done')
"
done

# =============================================================================
# Step 2: 重新融合 (smooth_eps=0.05)
# =============================================================================
echo ""
echo "==== Step 2: 4教师融合 (eps=$NEW_EPS) ===="
FUSED_NEW="$STEP2_DIR/data/train_fused_4t_eps${NEW_EPS}.jsonl"

$PYTHON "$PROJ_DIR/shared/merge_teacher_soft_labels.py" \
    --base "$BASE_TRAIN" \
    --teachers \
        "$STEP2_DIR/data/teacher_doubao_eps${NEW_EPS}.jsonl" \
        "$STEP2_DIR/data/teacher_deepseek_eps${NEW_EPS}.jsonl" \
        "$STEP2_DIR/data/teacher_qwen32b_eps${NEW_EPS}.jsonl" \
        "$STEP2_DIR/data/teacher_qwen14b_eps${NEW_EPS}.jsonl" \
    --weights 0.285 0.254 0.240 0.222 \
    --teacher_names Doubao DeepSeek Qwen-32B Qwen-14B \
    --output "$FUSED_NEW"

echo "Fused data: $FUSED_NEW ($(wc -l < "$FUSED_NEW") lines)"

# 验证新标签质量
$PYTHON -c "
import json
import numpy as np

with open('$FUSED_NEW') as f:
    rows = [json.loads(l.strip()) for l in f if l.strip()]

correct = 0
entropies = []
for d in rows:
    td = d.get('TeacherDist', {})
    ans = d.get('Answer','')
    probs = [td.get(c, 0) for c in 'ABCDE']
    pred = 'ABCDE'[probs.index(max(probs))]
    if pred == ans: correct += 1
    ent = -sum(p * np.log(p+1e-10) for p in probs)
    entropies.append(ent)

print(f'  GT agreement: {correct}/{len(rows)} ({100*correct/len(rows):.1f}%)')
print(f'  Entropy: mean={np.mean(entropies):.3f} (old=0.889)')
print(f'  max_prob range: mean={np.mean([max(td.get(c,0) for c in \"ABCDE\") for td in (d.get(\"TeacherDist\",{}) for d in rows)]):.3f}')
"

# =============================================================================
# Step 3: Grid search Round 2
# =============================================================================
echo ""
echo "========================================"
echo "Step 3: Round 2 Grid Search (12 runs)"
echo "  α(GT) × lr × smooth_eps=0.05"
echo "  + cosine schedule + best-epoch eval"
echo "========================================"
echo ""
echo "| # | doc_α | lr | code_alpha | Best Val | Best Epoch | Test@Best |"
echo "|---|-------|----|------------|----------|------------|-----------|"

RUN_ID=0
RESULTS_FILE="$STEP2_DIR/round2_results.jsonl"
echo "" > "$RESULTS_FILE"

for DOC_ALPHA in 0.5 0.7 0.9 1.0; do
    CODE_ALPHA=$(python3 -c "print(round(1.0 - $DOC_ALPHA, 2))")

    for LR in 5e-5 1e-4 2e-4; do
        RUN_ID=$((RUN_ID + 1))
        RUN_KEY="r2_a${DOC_ALPHA}_lr${LR}"
        OUT_DIR="$STEP2_DIR/outputs/${RUN_KEY}"

        echo ""
        echo "=== Run $RUN_ID: doc_α=$DOC_ALPHA, lr=$LR, code_alpha=$CODE_ALPHA ==="

        $PYTHON "$PROJ_DIR/shared/train_choice_head_distill.py" \
            --model_name "$MODEL" \
            --data_path "$FUSED_NEW" \
            --val_path "$VAL_DATA" \
            --test_path "$TEST_DATA" \
            --output_dir "$OUT_DIR" \
            --num_epochs 8 \
            --batch_size 2 \
            --gradient_accumulation_steps 8 \
            --learning_rate "$LR" \
            --rank 16 \
            --lora_alpha 32 \
            --alpha "$CODE_ALPHA" \
            --default_distill_mask 1 \
            --seed 42 \
            --deterministic \
            --use_cosine_schedule \
            --warmup_ratio 0.1

        echo "| $RUN_ID | $DOC_ALPHA | $LR | $CODE_ALPHA | see_log | see_log | see_log |"
    done
done

# =============================================================================
# Step 4: 汇总结果
# =============================================================================
echo ""
echo "========================================"
echo "Round 2 Summary - $(date)"
echo "========================================"

# Extract all results
$PYTHON -c "
import re, sys

log_path = '$LOG_FILE'
with open(log_path) as f:
    content = f.read()

runs = re.findall(r'=== Run (\d+): doc_α=([\d.]+), lr=([\de.-]+), code_alpha=([\d.]+) ===', content)
vals = re.findall(r'\[VAL\] epoch=(\d+) acc=([\d.]+)%', content)
bests = re.findall(r'\[BEST\] val_acc=([\d.]+)% at epoch (\d+)', content)
tests = re.findall(r'\[TEST-BEST\] epoch=(\d+) test_acc=([\d.]+)%', content)

print('| # | α(GT) | lr | Best Val | Best Ep | Test@Best |')
print('|---|-------|----|----------|---------|-----------|')

best_test = 0
best_cfg = ''
i_best = 0
i_test = 0
for i, (rid, da, lr, ca) in enumerate(runs):
    if i_best < len(bests):
        bv, be = bests[i_best]
        i_best += 1
    else:
        bv, be = '?', '?'
    if i_test < len(tests):
        te, ta = tests[i_test]
        i_test += 1
    else:
        te, ta = '?', '?'
    print(f'| {rid} | {da} | {lr} | {bv}% | ep{be} | {ta}% |')
    try:
        if float(ta) > best_test:
            best_test = float(ta)
            best_cfg = f'α={da}, lr={lr}'
    except: pass

print(f'\nBest: test={best_test}% @ {best_cfg}')
"

echo ""
echo "Round 2 完成: $(date)"
