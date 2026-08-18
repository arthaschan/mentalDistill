#!/usr/bin/env bash
# 中文牙科"学生超越老师"：Qwen3-32B 学生，α=0 纯 GT SFT，1 epoch，全量 4608 题训练 × 3 seed。
# 目标：训练后在 test_dental(125) 上超过 DeepSeek 老师 79.20%（3-seed 均值口径）。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # -> mentalDistill/
source setup.env 2>/dev/null || true
PY="${EASYEDIT_PY:-$HOME/anaconda3/bin/python3}"
RUN="fullEnglish/03_main_distill/runs"
STUDENT="models/Qwen3-32B"
DATA="15_fulldata_resplit/data"
SEEDS=(11 42 8)
export DISTILL_PROMPT_LANG=zh            # 中文牙科 prompt
export DISTILL_USE_CHAT_TEMPLATE=1      # Qwen3 自带模板 + 关 thinking

for seed in "${SEEDS[@]}"; do
  name="Qwen3_cn_a00_s${seed}"
  out="$RUN/$name"
  if [[ -f "$out/DONE" ]]; then echo "[SKIP] $name"; continue; fi
  mkdir -p "$out"
  echo "[$(date +%H:%M:%S)] TRAIN $name (alpha=0, 中文4608题 seed=$seed)"
  "$PY" shared/train_choice_head_distill.py \
    --model_name "$STUDENT" \
    --data_path "$DATA/train.jsonl" \
    --val_path "$DATA/val_dental.jsonl" \
    --output_dir "$out" --num_epochs 1 --batch_size 1 --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 --rank 16 --lora_alpha 32 --alpha 0.0 \
    --default_distill_mask 1 --seed "$seed" --deterministic \
    > "$out/train.log" 2>&1
  touch "$out/DONE"
  echo "[$(date +%H:%M:%S)] done $name"
done

echo "[$(date +%H:%M:%S)] Qwen3 中文牙科 3-seed 训练完成"
