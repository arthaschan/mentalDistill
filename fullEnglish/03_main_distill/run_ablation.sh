#!/usr/bin/env bash
# 学习预算消融（任务 4）：固定学生 Qwen2.5-32B，扫 LoRA rank / epoch，
# 看蒸馏增益是否在 ~4pp 饱和（检验"增益≈4pp 是任务固有 vs 训练预算封顶"）。
# 基线：rank16/1epoch（主实验 32B_a00，训练后 75.64%，增益 +4.26pp）。
# 配置：
#   rank64_s11 : rank 64, 1 epoch  （rank 是否瓶颈）
#   rank128_s11: rank 128, 1 epoch （rank 是否瓶颈，更强）
#   epoch3_s11 : rank 16, 3 epochs（epoch 是否瓶颈）
# 每个约 75-120 分钟，串行。val acc（MedQA dev）做快速代理，训练日志自带 [VAL]。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # -> mentalDistill/
source setup.env 2>/dev/null || true
PY="${EASYEDIT_PY:-$HOME/anaconda3/bin/python3}"
FE="fullEnglish/03_main_distill"
DATA="fullEnglish/00_data/out"
RUN="$FE/runs"
STUDENT="models/Qwen2.5-32B-Instruct"
export DISTILL_PROMPT_LANG=en   # Qwen2.5 硬编码模板，不设 DISTILL_USE_CHAT_TEMPLATE

wait_gpu_idle () {
  while :; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    if [[ -n "${used:-}" && "$used" -lt 20000 ]]; then return 0; fi
    echo "[$(date +%H:%M:%S)] GPU 忙 (${used:-?} MiB), 等待..."
    sleep 120
  done
}

run_arm () {
  local name="$1" rank="$2" epochs="$3"
  local out="$RUN/$name"
  if [[ -f "$out/DONE" ]]; then echo "[SKIP] $name"; return; fi
  mkdir -p "$out"
  wait_gpu_idle
  echo "-------------------------------------------------------------------"
  echo "[$(date +%H:%M:%S)] TRAIN $name (rank=$rank epochs=$epochs)"
  "$PY" shared/train_choice_head_distill.py \
    --model_name "$STUDENT" \
    --data_path "$FE/data/train_head_distill.jsonl" \
    --val_path "$DATA/val.jsonl" \
    --output_dir "$out" --num_epochs "$epochs" --batch_size 1 --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 --rank "$rank" --lora_alpha $((rank * 2)) --alpha 0.0 \
    --default_distill_mask 1 --seed 11 --deterministic \
    > "$out/train.log" 2>&1
  touch "$out/DONE"
  echo "[$(date +%H:%M:%S)] done $name"
}

run_arm "32B_ab_rank64_s11"  64 1
run_arm "32B_ab_rank128_s11" 128 1
run_arm "32B_ab_epoch3_s11"  16 3

echo "[$(date +%H:%M:%S)] 学习预算消融完成"
