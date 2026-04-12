#!/usr/bin/env bash
# 直接运行 Doubao 多票采样
set -euo pipefail
cd /home/student/arthas/mentalDistill
source setup.env

echo "[START] Doubao multivote: $(date)"
echo "DOUBAO_API_KEY length: ${#DOUBAO_API_KEY}"

python3 -u shared/generate_teacher_soft_labels_multivote.py \
  --existing_labels 03_doubao_choice_head/data/teacher_train.jsonl \
  --candidate 03_doubao_choice_head/configs/teacher_candidate.json \
  --system_prompt shared/system_prompt_mcq.txt \
  --output 03_doubao_choice_head/data/teacher_train_multivote.jsonl \
  --extra_votes 8 \
  --temperature 0.9 \
  --timeout_sec 120 \
  --max_retries 4 \
  --request_interval_sec 0.3 \
  --cooldown_every 50 \
  --cooldown_sec 5.0 \
  --resume

echo "[DONE] Doubao multivote: $(date)"
