#!/usr/bin/env bash
# 27 英文全科·无印度：Qwen2.5-32B 学生，α=0 纯 GT，1 epoch，训练集去掉 MedMCQA(印度)。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."   # -> 27_english_general_noindia/
source ../setup.env 2>/dev/null || true
PY="${EASYEDIT_PY:-$HOME/anaconda3/bin/python3}"
STUDENT="../models/Qwen2.5-32B-Instruct"
export DISTILL_PROMPT_LANG=en
export DISTILL_USE_CHAT_TEMPLATE=1

mkdir -p runs
name="32B_noindia_a00_s42"
out="runs/$name"
if [[ -f "$out/DONE" ]]; then echo "[SKIP] $name"; exit 0; fi
mkdir -p "$out"
echo "[$(date +%H:%M:%S)] TRAIN $name (无印度 10168 题[牙科重划], seed=42)"
"$PY" ../shared/train_choice_head_distill.py \
  --model_name "$STUDENT" \
  --data_path data/train_no_india_dentalsplit.jsonl \
  --output_dir "$out" --num_epochs 1 --batch_size 1 --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 --rank 16 --lora_alpha 32 --alpha 0.0 \
  --default_distill_mask 1 --seed 42 --deterministic \
  > "$out/train.log" 2>&1
touch "$out/DONE"
echo "[$(date +%H:%M:%S)] done $name"
