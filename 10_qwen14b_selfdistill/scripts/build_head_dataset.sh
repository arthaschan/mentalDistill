#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/../shared/common_env.sh"
resolve_python

mkdir -p "$ROOT_DIR/logs"

"$PY" "$SHARED_DIR/build_selective_distill_dataset.py" \
  --gt_data "$ROOT_DIR/data/train.jsonl" \
  --teacher_soft "$ROOT_DIR/data/teacher_train_soft.jsonl" \
  --output "$ROOT_DIR/data/train_head_distill.jsonl" \
  --report "$ROOT_DIR/data/distill_dataset_report.txt" \
  --min_entropy 0.20 \
  --smooth_eps 0.25 \
  --min_margin 0.03 \
  2>&1 | tee "$ROOT_DIR/logs/build_head_dataset.log"
