#!/usr/bin/env bash
set -euo pipefail
# Full pipeline for Module 15: full-data resplit experiments
# Steps: teacher labels → soft labels → distill dataset → train → eval

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=============================================="
echo "Module 15: Full-Data Resplit Pipeline"
echo "Train: 4608  Val: 991  Test: 991"
echo "Dental subset: train=580 val=125 test=125"
echo "=============================================="

echo ""
echo ">>> Step 1/6: Generate DeepSeek teacher labels"
bash "$ROOT_DIR/scripts/generate_teacher_labels.sh"

echo ""
echo ">>> Step 2/6: Prepare soft labels"
bash "$ROOT_DIR/scripts/prepare_soft_labels.sh"

echo ""
echo ">>> Step 3/6: Build distillation dataset"
bash "$ROOT_DIR/scripts/build_head_dataset.sh"

echo ""
echo ">>> Step 4/6: Train 7B student (3 seeds, two-stage)"
bash "$ROOT_DIR/scripts/run_train_7b.sh"

echo ""
echo ">>> Step 5/6: Train 14B student (3 seeds, stage1-only)"
bash "$ROOT_DIR/scripts/run_train_14b.sh"

echo ""
echo ">>> Step 6/6: Dual evaluation (full + dental)"
# Find latest 7B and 14B runs
LATEST_7B="$(ls -td "$ROOT_DIR"/runs/*_fulldata_7b 2>/dev/null | head -1)"
LATEST_14B="$(ls -td "$ROOT_DIR"/runs/*_fulldata_14b 2>/dev/null | head -1)"

if [[ -n "$LATEST_7B" ]]; then
  echo "Evaluating 7B: $LATEST_7B"
  python3 "$ROOT_DIR/scripts/run_eval_dual.py" \
    --run_root "$LATEST_7B" \
    --student_size 7b
fi

if [[ -n "$LATEST_14B" ]]; then
  echo "Evaluating 14B: $LATEST_14B"
  python3 "$ROOT_DIR/scripts/run_eval_dual.py" \
    --run_root "$LATEST_14B" \
    --student_size 14b
fi

echo ""
echo "=============================================="
echo "Pipeline complete!"
echo "=============================================="
