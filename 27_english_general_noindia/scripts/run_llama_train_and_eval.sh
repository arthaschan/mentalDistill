#!/usr/bin/env bash
# Llama-70B 无印度：训练 → 全科评测 → 牙科评测 串联（训练在后台跑，本脚本等 DONE 后自动接评测）。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"   # -> 27_english_general_noindia/scripts
cd ..
PY="${EASYEDIT_PY:-$HOME/anaconda3/bin/python3}"

train_done="runs/Llama70B_noindia_a00_s42/DONE"
echo "[$(date +%H:%M:%S)] 等待 Llama-70B 训练完成 (DONE) ..."
while [[ ! -f "$train_done" ]]; do sleep 30; done
echo "[$(date +%H:%M:%S)] 训练完成，开始全科评测 (4110 题)"

"$PY" scripts/eval_noindia_full_llama.py > runs/eval_full_llama.log 2>&1
echo "[$(date +%H:%M:%S)] 全科评测完成，开始牙科评测 (501 题)"

cd ../28_english_dental_noindia
"$PY" scripts/eval_noindia_dental_llama.py > runs/eval_dental_llama.log 2>&1

echo "[$(date +%H:%M:%S)] Llama-70B 无印度 全科+牙科评测全部完成"
