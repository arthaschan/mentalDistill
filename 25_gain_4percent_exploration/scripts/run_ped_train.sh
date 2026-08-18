#!/usr/bin/env bash
# PED 训练：只训"差点答对"(near-miss) 题（2904 道），α=0 纯 GT SFT，3 seed。
# 预期：训练后学生在全量测试集塌到 ~15.42%（比瞎猜还低）——证明 4% 抬不上去。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."   # -> 24_gain_4percent_exploration/
source ../setup.env 2>/dev/null || true
PY="${EASYEDIT_PY:-$HOME/anaconda3/bin/python3}"
STUDENT="../models/Qwen3-32B"
DATA="data"
VAL="../fullEnglish/00_data/out/val.jsonl"
RUN="runs"
SEEDS=(11 42 8)
export DISTILL_PROMPT_LANG=en
export DISTILL_USE_CHAT_TEMPLATE=1

mkdir -p "$RUN"
for seed in "${SEEDS[@]}"; do
  name="Qwen3_ped_ar_s${seed}"
  out="$RUN/$name"
  if [[ -f "$out/DONE" ]]; then echo "[SKIP] $name"; continue; fi
  mkdir -p "$out"
  echo "[$(date +%H:%M:%S)] TRAIN $name (alpha=0 seed=$seed)"
  "$PY" ../shared/train_choice_head_distill.py \
    --model_name "$STUDENT" \
    --data_path "$DATA/train_head_almostright.jsonl" \
    --val_path "$VAL" \
    --output_dir "$out" --num_epochs 1 --batch_size 1 --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 --rank 16 --lora_alpha 32 --alpha 0.0 \
    --default_distill_mask 1 --seed "$seed" --deterministic \
    > "$out/train.log" 2>&1
  touch "$out/DONE"
  echo "[$(date +%H:%M:%S)] done $name"
done

echo "[$(date +%H:%M:%S)] PED 3-seed 训练完成"
