#!/usr/bin/env bash
# Diagnostic: UK/US-only training (447 items, NO MedMCQA) at alpha=0, to isolate the
# distribution-mismatch confound. Compare test_ukus vs the mixed-pool alpha=0 arms.
# Same hyperparams/seeds as main run; ONLY the training set differs.
set -uo pipefail
cd /home/student/arthas/mentalDistill
source setup.env 2>/dev/null || true
PY="${EASYEDIT_PY:-/home/student/anaconda3/bin/python3}"
SHARED=shared
DATA=english/03_main_distill/data
RUN=english/03_main_distill/runs_diag
mkdir -p "$RUN"

TRAIN="$DATA/train_head_ukus_only.jsonl"
VAL="$DATA/val.jsonl"; TEST="$DATA/test_ukus.jsonl"; TEST2="$DATA/test_medmcqa.jsonl"
declare -A MODEL=( [7B]="$BASE_MODEL_7B" [14B]="$BASE_MODEL_14B" )
declare -A LR=( [7B]="0.00012" [14B]="0.0001" )
SEEDS=(11 42 8)

for size in 7B 14B; do
  for seed in "${SEEDS[@]}"; do
    name="${size}_ukusonly_a00_s${seed}"; out="$RUN/$name"
    [[ -f "$out/DONE" ]] && { echo "[SKIP] $name"; continue; }
    mkdir -p "$out"
    echo "[$(date +%H:%M:%S)] TRAIN $name (UK/US-only 447, alpha=0)"
    "$PY" "$SHARED/train_choice_head_distill.py" \
      --model_name "${MODEL[$size]}" --data_path "$TRAIN" --val_path "$VAL" --test_path "$TEST" \
      --output_dir "$out" --num_epochs 1 --batch_size 2 --gradient_accumulation_steps 4 \
      --learning_rate "${LR[$size]}" --rank 16 --lora_alpha 32 --alpha 0.0 \
      --default_distill_mask 0 --seed "$seed" --deterministic > "$out/train.log" 2>&1
    [[ -d "$out/best" ]] && "$PY" "$SHARED/evaluate_model.py" --base_model "${MODEL[$size]}" \
      --adapter_dir "$out/best" --test_data "$TEST2" > "$out/eval_medmcqa.log" 2>&1 || true
    touch "$out/DONE"; echo "[$(date +%H:%M:%S)] done $name"
  done
done
echo "[$(date +%H:%M:%S)] DIAGNOSTIC COMPLETE"
