#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/../shared/common_env.sh"
resolve_python
resolve_model_dir BASE_MODEL_7B Qwen2.5-7B-Instruct
RUN_ROOT="$ROOT_DIR/runs/$(date +%Y%m%d_%H%M%S)_q14self_best"

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/outputs"

"$PY" "$SHARED_DIR/run_two_stage_training.py" \
  --params "$ROOT_DIR/configs/grid_params_best.json" \
  --run_root "$RUN_ROOT" \
  --project_root "$SHARED_DIR/.." \
  --base_model "$BASE_MODEL_7B" \
  --train_head "$ROOT_DIR/data/train_head_distill.jsonl" \
  --train_gt "$ROOT_DIR/data/train.jsonl" \
  --val_data "$ROOT_DIR/data/val.jsonl" \
  --test_data "$ROOT_DIR/data/test.jsonl" \
  --teacher_prefix q14self \
  --py "$PY" \
  2>&1 | tee "$RUN_ROOT/logs/two_stage.log"
