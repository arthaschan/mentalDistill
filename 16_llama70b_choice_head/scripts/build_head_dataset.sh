#!/usr/bin/env bash
set -euo pipefail
# Build distill dataset from Llama-70B real logprobs teacher labels.
# Since vLLM gives real probability distributions, use smooth_eps=0 (no artificial smoothing)
# and very low min_entropy/min_margin to include nearly all samples.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/../shared/common_env.sh"
resolve_python

mkdir -p "$ROOT_DIR/logs"

"$PY" "$SHARED_DIR/build_selective_distill_dataset.py" \
  --gt_data "$ROOT_DIR/../15_fulldata_resplit/data/train.jsonl" \
  --teacher_soft "$ROOT_DIR/data/teacher_train.jsonl" \
  --output "$ROOT_DIR/data/train_head_distill.jsonl" \
  --report "$ROOT_DIR/data/distill_dataset_report.txt" \
  --min_entropy 0.01 \
  --smooth_eps 0.0 \
  --min_margin 0.0 \
  2>&1 | tee "$ROOT_DIR/logs/build_head_dataset.log"
