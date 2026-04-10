#!/usr/bin/env bash
set -euo pipefail

echo "rebuild 目录包含以下实验:"
for d in 00_baseline_gt_sft 01_qwen32b_whitebox 02_deepseek_v3_choice_head 03_doubao_choice_head 04_kimi_choice_head 05_qwen14b_choice_head; do
  echo ""
  echo "=== $d ==="
  if [[ -f "$d/README.md" ]]; then
    sed -n '1,80p' "$d/README.md"
  else
    echo "missing README.md"
  fi
done
