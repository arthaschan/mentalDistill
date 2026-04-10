#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/../shared/common_env.sh"
resolve_python
resolve_model_dir BASE_MODEL_7B Qwen2.5-7B-Instruct
resolve_model_dir BASE_MODEL_32B Qwen2.5-32B-Instruct

cd "$ROOT_DIR"
mkdir -p outputs logs

"$PY" "$SHARED_DIR/train_whitebox_distill.py" \
  --teacher_model_name "$BASE_MODEL_32B" \
  --student_model_name "$BASE_MODEL_7B" \
  --data_path "$ROOT_DIR/data/train.jsonl" \
  --val_path "$ROOT_DIR/data/val.jsonl" \
  --test_path "$ROOT_DIR/data/test.jsonl" \
  --output_dir "$ROOT_DIR/outputs" \
  --num_epochs 4 \
  --batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 \
  --rank 16 \
  --lora_alpha 32 \
  --temperature 2.0 \
  --alpha 0.1 \
  --alpha_warmup_epochs 1 \
  --hard_upsample 2 \
  --seed 2 \
  2>&1 | tee "$ROOT_DIR/logs/train.log"
