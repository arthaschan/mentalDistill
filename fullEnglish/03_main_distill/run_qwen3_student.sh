#!/usr/bin/env bash
# fullEnglish — Qwen3-32B 学生蒸馏（α=0 纯 GT SFT × 3 seed）。
# 与主线 32B 同配置（batch1 accum8 / lr1e-4 / rank16 / 1 epoch），仅换 base 模型 + chat template。
# Qwen3 是混合 thinking 模型：必须 DISTILL_USE_CHAT_TEMPLATE=1（用自带模板 + 关 thinking），
# 否则硬编码 Qwen2.5 模板 + eos 错会触发推理链 → 训练/eval 全废（见工作记录 08-16）。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # -> mentalDistill/
source setup.env 2>/dev/null || true
PY="${EASYEDIT_PY:-$HOME/anaconda3/bin/python3}"
FE="fullEnglish/03_main_distill"
DATA="fullEnglish/00_data/out"
RUN="$FE/runs"
STUDENT="models/Qwen3-32B"
SEEDS=(11 42 8)
export DISTILL_PROMPT_LANG=en
export DISTILL_USE_CHAT_TEMPLATE=1   # Qwen3 用自带 chat template（含 enable_thinking 关闭逻辑）

wait_gpu_idle () {
  while :; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    if [[ -n "${used:-}" && "$used" -lt 20000 ]]; then return 0; fi
    echo "[$(date +%H:%M:%S)] GPU 忙 (${used:-?} MiB), 等待其他训练释放..."
    sleep 120
  done
}

mkdir -p "$RUN"
if [[ ! -d "$STUDENT" ]]; then echo "[FATAL] 学生模型缺失: $STUDENT"; exit 1; fi
if [[ ! -f "$FE/data/train_head_distill.jsonl" ]]; then echo "[FATAL] 训练数据缺失，先跑 build_train_head.py"; exit 1; fi

echo "==================================================================="
echo "Qwen3-32B 学生蒸馏 — train=$(wc -l < "$FE/data/train_head_distill.jsonl")  val=$(wc -l < "$DATA/val.jsonl")"
echo "==================================================================="

for seed in "${SEEDS[@]}"; do
  name="Qwen3_a00_s${seed}"
  out="$RUN/$name"
  if [[ -f "$out/DONE" ]]; then echo "[SKIP] $name"; continue; fi
  mkdir -p "$out"
  wait_gpu_idle
  echo "-------------------------------------------------------------------"
  echo "[$(date +%H:%M:%S)] TRAIN $name (alpha=0 seed=$seed)"
  "$PY" shared/train_choice_head_distill.py \
    --model_name "$STUDENT" \
    --data_path "$FE/data/train_head_distill.jsonl" \
    --val_path "$DATA/val.jsonl" \
    --output_dir "$out" --num_epochs 1 --batch_size 1 --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 --rank 16 --lora_alpha 32 --alpha 0.0 \
    --default_distill_mask 1 --seed "$seed" --deterministic \
    > "$out/train.log" 2>&1
  touch "$out/DONE"
  echo "[$(date +%H:%M:%S)] done $name"
done

echo "[$(date +%H:%M:%S)] Qwen3-32B 3-seed 训练完成"
