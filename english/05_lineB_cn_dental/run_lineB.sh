#!/usr/bin/env bash
# Paper Line B: Chinese dental-specialist student. Can it beat teacher (86.4%) on the
# 125-question dental subset that the original paper did NOT beat?
# alpha=0 headline (per ablation) + alpha=0.35 comparison, x {7B,14B} x 3 seeds.
set -uo pipefail
cd /home/student/arthas/mentalDistill
source setup.env 2>/dev/null || true
PY="${EASYEDIT_PY:-/home/student/anaconda3/bin/python3}"
SHARED=shared
DATA=english/05_lineB_cn_dental/data
RUN=english/05_lineB_cn_dental/runs
mkdir -p "$RUN"
TRAIN="$DATA/train_head_distill.jsonl"
VAL="$DATA/val_dental.jsonl"; TEST="$DATA/test_dental.jsonl"
declare -A MODEL=( [7B]="$BASE_MODEL_7B" [14B]="$BASE_MODEL_14B" )
declare -A LR=( [7B]="0.00012" [14B]="0.0001" )
SEEDS=(11 42 8)

run_arm(){
  local size=$1 alpha=$2 tag=$3
  for seed in "${SEEDS[@]}"; do
    local name="${size}_${tag}_s${seed}"; local out="$RUN/$name"
    [[ -f "$out/DONE" ]] && { echo "[SKIP] $name"; continue; }
    mkdir -p "$out"
    echo "[$(date +%H:%M:%S)] TRAIN $name (alpha=$alpha, CN dental 4990)"
    "$PY" "$SHARED/train_choice_head_distill.py" \
      --model_name "${MODEL[$size]}" --data_path "$TRAIN" --val_path "$VAL" --test_path "$TEST" \
      --output_dir "$out" --num_epochs 1 --batch_size 2 --gradient_accumulation_steps 4 \
      --learning_rate "${LR[$size]}" --rank 16 --lora_alpha 32 --alpha "$alpha" \
      --default_distill_mask 0 --seed "$seed" --deterministic > "$out/train.log" 2>&1
    touch "$out/DONE"; echo "[$(date +%H:%M:%S)] done $name"
  done
}
echo "########## STEP 0: zero-shot floor on 125 dental ##########"
for size in 7B 14B; do
  "$PY" "$SHARED/evaluate_model.py" --base_model "${MODEL[$size]}" --test_data "$TEST" \
    > "$RUN/zeroshot_${size}.log" 2>&1 || true
done
echo "########## STEP 1: alpha=0 headline ##########"
run_arm 7B 0.0 a00
run_arm 14B 0.0 a00
echo "########## STEP 2: alpha=0.35 comparison ##########"
run_arm 7B 0.35 a35
run_arm 14B 0.35 a35
echo "[$(date +%H:%M:%S)] LINE B COMPLETE"
