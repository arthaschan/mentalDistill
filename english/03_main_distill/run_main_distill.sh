#!/usr/bin/env bash
# Main distillation experiment (overnight, single H100, SEQUENTIAL).
# Teacher = DeepSeek-V3 (82.86% on English dental). Student = Qwen2.5-7B / 14B.
# Design (single-variable): alpha=0.35 (distill) vs alpha=0.0 (GT-only null control), x3 seeds.
# Selection on val (UK/US), report on test_ukus (primary) + test_medmcqa (cross-source).
set -uo pipefail
cd /home/student/arthas/mentalDistill
source setup.env 2>/dev/null || true
PY="${EASYEDIT_PY:-/home/student/anaconda3/bin/python3}"
SHARED=shared
DATA=english/03_main_distill/data
RUN=english/03_main_distill/runs
mkdir -p "$RUN"

TRAIN="$DATA/train_head_distill.jsonl"
VAL="$DATA/val.jsonl"
TEST="$DATA/test_ukus.jsonl"
TEST2="$DATA/test_medmcqa.jsonl"

declare -A MODEL=( [7B]="$BASE_MODEL_7B" [14B]="$BASE_MODEL_14B" )
declare -A LR=( [7B]="0.00012" [14B]="0.0001" )
SEEDS=(11 42 8)

run_arm () {
  local size=$1 alpha=$2 tag=$3
  local mp="${MODEL[$size]}" lr="${LR[$size]}"
  for seed in "${SEEDS[@]}"; do
    local name="${size}_${tag}_s${seed}"
    local out="$RUN/$name"
    if [[ -f "$out/DONE" ]]; then echo "[SKIP] $name"; continue; fi
    mkdir -p "$out"
    echo "=================================================================="
    echo "[$(date +%H:%M:%S)] TRAIN $name  (alpha=$alpha lr=$lr model=$size)"
    "$PY" "$SHARED/train_choice_head_distill.py" \
      --model_name "$mp" --data_path "$TRAIN" --val_path "$VAL" --test_path "$TEST" \
      --output_dir "$out" --num_epochs 1 --batch_size 2 --gradient_accumulation_steps 4 \
      --learning_rate "$lr" --rank 16 --lora_alpha 32 --alpha "$alpha" \
      --default_distill_mask 0 --seed "$seed" --deterministic \
      > "$out/train.log" 2>&1
    # cross-source eval on the best adapter
    if [[ -d "$out/best" ]]; then
      "$PY" "$SHARED/evaluate_model.py" --base_model "$mp" --adapter_dir "$out/best" \
        --test_data "$TEST2" > "$out/eval_medmcqa.log" 2>&1 || true
    fi
    touch "$out/DONE"
    echo "[$(date +%H:%M:%S)] done $name"
  done
}

echo "########## STEP 0: build training data with DeepSeek soft labels ##########"
"$PY" english/03_main_distill/build_train_head.py 2>&1 | tee "$RUN/build_data.log"

echo "########## STEP 1: zero-shot student floor (no training) ##########"
for size in 7B 14B; do
  "$PY" "$SHARED/evaluate_model.py" --base_model "${MODEL[$size]}" \
    --test_data "$TEST" > "$RUN/zeroshot_${size}_ukus.log" 2>&1 || true
done

# alpha sweep to test if the Chinese ablation finding (alpha=0 best, KL monotonically hurts)
# REPLICATES on English dental. alpha=0.0 is the HEADLINE arm (per prior ablation).
echo "########## STEP 2: alpha=0.0 (HEADLINE: pure CE / decision-space supervision) ##########"
run_arm 7B 0.0 a00
run_arm 14B 0.0 a00
echo "########## STEP 3: alpha=0.35 (KL-distill comparison, old main setting) ##########"
run_arm 7B 0.35 a35
run_arm 14B 0.35 a35
echo "########## STEP 4: alpha=1.0 (pure KL / imitate teacher, worst-case anchor) ##########"
run_arm 7B 1.0 a10
run_arm 14B 1.0 a10

echo "########## STEP 4: aggregate ##########"
"$PY" english/03_main_distill/aggregate_results.py 2>&1 | tee "$RUN/RESULTS.log"
echo "[$(date +%H:%M:%S)] MAIN DISTILL COMPLETE"
