#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/../shared/common_env.sh"
resolve_python
resolve_model_dir BASE_MODEL_7B Qwen2.5-7B-Instruct

cd "$ROOT_DIR"
mkdir -p outputs logs

"$PY" "$SHARED_DIR/train_gt_sft.py" \
  --model_name "$BASE_MODEL_7B" \
  --data_path "$ROOT_DIR/data/train.jsonl" \
  --val_path "$ROOT_DIR/data/val.jsonl" \
  --test_path "$ROOT_DIR/data/test.jsonl" \
  --output_dir "$ROOT_DIR/outputs" \
  --num_epochs 5 \
  --batch_size 4 \
  --gradient_accumulation_steps 1 \
  --learning_rate 2e-4 \
  --rank 16 \
  --lora_alpha 32 \
  --alpha 0.0 \
  --seed 42 \
  --deterministic \
  2>&1 | tee "$ROOT_DIR/logs/train.log"
