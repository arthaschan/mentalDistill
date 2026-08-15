#!/usr/bin/env bash
# fullEnglish — Llama-3.3-70B 美国学生对照 (QLoRA, α=0 × 3 seed, 教师 DeepSeek).
# 学生 = Llama-3.3-70B-Instruct (Meta 美国模型), 4bit QLoRA 训练 (bf16 装不下 95GB).
# 用 tokenizer 自带 chat template (DISTILL_USE_CHAT_TEMPLATE=1).
# GPU 互斥, 与其它训练串行.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source setup.env 2>/dev/null || true
PY="${EASYEDIT_PY:-$HOME/anaconda3/bin/python3}"
FE="fullEnglish/03_main_distill"
DATA="fullEnglish/00_data/out"
RUN="$FE/runs"
STUDENT="${LLAMA70B_MODEL:-models/Llama-3.3-70B-Instruct}"
SEEDS=(11 42 8)
export DISTILL_PROMPT_LANG=en
export DISTILL_USE_CHAT_TEMPLATE=1

TRAIN_FILE="$FE/data/train_head_distill.jsonl"

if [[ ! -d "$STUDENT" ]]; then
  echo "[FATAL] Llama70B 学生模型缺失: $STUDENT"
  exit 1
fi
if [[ ! -f "$TRAIN_FILE" ]]; then
  echo "[FATAL] 训练文件缺失: $TRAIN_FILE (先跑 run_main_distill.sh 的 STEP 1)"
  exit 1
fi

wait_gpu_idle () {
  while :; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    if [[ -n "${used:-}" && "$used" -lt 20000 ]]; then return 0; fi
    echo "[$(date +%H:%M:%S)] GPU 忙 (${used:-?} MiB), 等待其他任务释放..."
    sleep 120
  done
}

echo "=== 训练 Llama-3.3-70B 学生 (QLoRA α=0 × ${#SEEDS[@]} seed, DeepSeek 教师) ==="
for seed in "${SEEDS[@]}"; do
  name="Llama70B_a00_s${seed}"
  out="$RUN/$name"
  if [[ -f "$out/DONE" ]]; then echo "[SKIP] $name"; continue; fi
  mkdir -p "$out"
  wait_gpu_idle
  echo "[$(date +%H:%M:%S)] TRAIN $name (Llama70B QLoRA α=0 seed=$seed)"
  "$PY" shared/train_choice_head_distill.py \
    --model_name "$STUDENT" \
    --data_path "$TRAIN_FILE" \
    --val_path "$DATA/val.jsonl" \
    --output_dir "$out" --num_epochs 1 --batch_size 1 --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 --rank 16 --lora_alpha 32 --alpha 0.0 \
    --default_distill_mask 1 --seed "$seed" --deterministic \
    --quantize 4bit \
    > "$out/train.log" 2>&1
  touch "$out/DONE"
  echo "[$(date +%H:%M:%S)] done $name"
done
echo "[$(date +%H:%M:%S)] Llama70B 学生训练全部完成"
