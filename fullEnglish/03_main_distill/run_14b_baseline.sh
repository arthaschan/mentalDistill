#!/usr/bin/env bash
# fullEnglish — 14B 学生对照训练 (消融臂: 学生规模 32B vs 14B, 同教师 DeepSeek).
# 自动: 等主线训练文件(DeepSeek 标签)生成 -> 等 GPU 空闲 -> 训练 14B α=0 × 3 seed.
# GPU 互斥, 排在 32B 主线之后, 不抢卡.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source setup.env 2>/dev/null || true
PY="${EASYEDIT_PY:-$HOME/anaconda3/bin/python3}"
FE="fullEnglish/03_main_distill"
DATA="fullEnglish/00_data/out"
RUN="$FE/runs"
STUDENT="${BASE_MODEL_14B:-models/Qwen2.5-14B-Instruct}"
SEEDS=(11 42 8)
export DISTILL_PROMPT_LANG=en

TRAIN_FILE="$FE/data/train_head_distill.jsonl"

echo "=== 14B 对照: 等待主线训练文件 (DeepSeek 标签 build 完成) ==="
waited=0
while [[ ! -f "$TRAIN_FILE" ]]; do
  sleep 120; waited=$((waited+120))
  echo "[$(date +%H:%M:%S)] 等待 $TRAIN_FILE (${waited}s)"
  if [[ $waited -ge 21600 ]]; then echo "[FATAL] 超时(6h)"; exit 1; fi
done
echo "训练文件就绪"

wait_gpu_idle () {
  while :; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    if [[ -n "${used:-}" && "$used" -lt 20000 ]]; then
      # 二次确认: 空闲后等 180s 再查一次, 让主线(32B, 优先)先抢 GPU, 避免竞态 OOM
      sleep 180
      used2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
      if [[ -n "${used2:-}" && "$used2" -lt 20000 ]]; then return 0; fi
      continue
    fi
    sleep 120
  done
}

echo "=== 训练 14B α=0 × 3 seed (DeepSeek 教师) ==="
for seed in "${SEEDS[@]}"; do
  name="14B_a00_s${seed}"
  out="$RUN/$name"
  if [[ -f "$out/DONE" ]]; then echo "[SKIP] $name"; continue; fi
  mkdir -p "$out"
  wait_gpu_idle
  echo "[$(date +%H:%M:%S)] TRAIN $name (14B 学生, DeepSeek 教师, α=0)"
  "$PY" shared/train_choice_head_distill.py \
    --model_name "$STUDENT" \
    --data_path "$TRAIN_FILE" \
    --val_path "$DATA/val.jsonl" \
    --output_dir "$out" --num_epochs 1 --batch_size 2 --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 --rank 16 --lora_alpha 32 --alpha 0.0 \
    --default_distill_mask 1 --seed "$seed" --deterministic \
    > "$out/train.log" 2>&1
  touch "$out/DONE"
  echo "[$(date +%H:%M:%S)] done $name"
done
echo "[$(date +%H:%M:%S)] 14B 对照训练完成"
