#!/usr/bin/env bash
# fullEnglish — Llama70B 备选教师臂 (自动流水线).
# 等 Llama70B 标签完成 -> build_train_head -> 训练 32B α=0 × 3 seed.
# 训练前用 GPU 空闲检测互斥, 与 DeepSeek 主线(32B_a00_*) 自动串行, 不抢卡.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source setup.env 2>/dev/null || true
PY="${EASYEDIT_PY:-$HOME/anaconda3/bin/python3}"
FE="fullEnglish/03_main_distill"
DATA="fullEnglish/00_data/out"
RUN="$FE/runs"
STUDENT="${STUDENT_MODEL:-$BASE_MODEL_32B}"
STUDENT="${STUDENT:-models/Qwen2.5-32B-Instruct}"
SEEDS=(11 42 8)
export DISTILL_PROMPT_LANG=en

LABEL="$FE/labels/teacher_train_llama70b.jsonl"
TARGET=$(wc -l < "$DATA/train.jsonl")

echo "=== Llama70B 臂: 等待标签完成 (目标 $TARGET 行) ==="
waited=0
while [[ ! -f "$LABEL" || $(wc -l < "$LABEL") -lt $TARGET ]]; do
  sleep 60; waited=$((waited+60))
  cur=$(wc -l < "$LABEL" 2>/dev/null || echo 0)
  echo "[$(date +%H:%M:%S)] 等待标签 $cur/$TARGET"
  if [[ $waited -ge 10800 ]]; then echo "[FATAL] 等标签超时(3h), 检查 vLLM 进程"; exit 1; fi
done
echo "标签完成: $(wc -l < "$LABEL") 行"

echo "=== build_train_head (Llama70B 真实 logprobs) ==="
"$PY" "$FE/build_train_head.py" --train "$DATA/train.jsonl" --teacher "$LABEL" \
    --output "$FE/data/train_head_distill_llama70b.jsonl"

wait_gpu_idle () {
  while :; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    if [[ -n "${used:-}" && "$used" -lt 20000 ]]; then return 0; fi
    echo "[$(date +%H:%M:%S)] GPU 忙 (${used:-?} MiB), 等待..."
    sleep 60
  done
}

echo "=== 训练 32B α=0 × 3 seed (Llama70B 教师) ==="
for seed in "${SEEDS[@]}"; do
  name="32B_llama70b_a00_s${seed}"
  out="$RUN/$name"
  if [[ -f "$out/DONE" ]]; then echo "[SKIP] $name"; continue; fi
  mkdir -p "$out"
  wait_gpu_idle
  echo "[$(date +%H:%M:%S)] TRAIN $name (Llama70B 教师, α=0)"
  "$PY" shared/train_choice_head_distill.py \
    --model_name "$STUDENT" \
    --data_path "$FE/data/train_head_distill_llama70b.jsonl" \
    --val_path "$DATA/val.jsonl" \
    --output_dir "$out" --num_epochs 1 --batch_size 1 --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 --rank 16 --lora_alpha 32 --alpha 0.0 \
    --default_distill_mask 1 --seed "$seed" --deterministic \
    > "$out/train.log" 2>&1
  touch "$out/DONE"
  echo "[$(date +%H:%M:%S)] done $name"
done
echo "[$(date +%H:%M:%S)] Llama70B 臂训练完成"
