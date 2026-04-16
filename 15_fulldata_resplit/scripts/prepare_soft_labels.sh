#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/../shared/common_env.sh"
resolve_python

mkdir -p "$ROOT_DIR/logs"

"$PY" "$SHARED_DIR/prepare_soft_labels.py" \
  --input "$ROOT_DIR/data/teacher_train.jsonl" \
  --output "$ROOT_DIR/data/teacher_train_soft.jsonl" \
  --smooth_eps 0.25 \
  2>&1 | tee "$ROOT_DIR/logs/prepare_soft_labels.log"
