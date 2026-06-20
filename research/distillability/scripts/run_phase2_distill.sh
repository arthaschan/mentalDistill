#!/usr/bin/env bash
set -euo pipefail
# run_phase2_distill.sh <teacher_label>
# Phase-2 distillation VALIDATION for one teacher (run ONLY after its predictions.json freeze).
# Builds teacher distill dataset from real logprobs, then 3-arm (baseline/geom/random) like Task A,
# so we can test H2: does (geom - random) gain scale with the teacher's geom_auc?
#
# Usage: SEED=42 bash research/distillability/scripts/run_phase2_distill.sh phi-4

LABEL="${1:?usage: run_phase2_distill.sh <teacher_label>}"
MODULE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # research/distillability
REPO_ROOT="$(cd "$MODULE_DIR/../.." && pwd)"
ROOT_DIR="$REPO_ROOT/research"   # so common_env's REBUILD_ROOT = repo root
source "$REPO_ROOT/shared/common_env.sh"
resolve_python
resolve_model_dir BASE_MODEL_14B Qwen2.5-14B-Instruct

DATA15="$REPO_ROOT/15_fulldata_resplit/data"
LOGPROBS="$MODULE_DIR/teacher_labels/${LABEL}_train_logprobs.jsonl"
DS_DIR="$MODULE_DIR/datasets/${LABEL}"
RUN_ROOT="$MODULE_DIR/runs/phase2_${LABEL}"
SEED="${SEED:-42}"
mkdir -p "$DS_DIR" "$RUN_ROOT/logs"

[ -f "$LOGPROBS" ] || { echo "[ABORT] missing logprobs: $LOGPROBS"; exit 2; }

# 1. build the head-distill dataset from REAL logprobs (no artificial smoothing; keep real structure)
HEAD="$DS_DIR/train_head_distill.jsonl"
if [ ! -f "$HEAD" ]; then
  echo "=== build head-distill dataset for $LABEL ==="
  "$PY" "$SHARED_DIR/build_selective_distill_dataset.py" \
    --gt_data "$DATA15/train.jsonl" \
    --teacher_soft "$LOGPROBS" \
    --output "$HEAD" \
    --report "$DS_DIR/distill_dataset_report.txt" \
    --min_entropy 0.01 --smooth_eps 0.0 --min_margin 0.0
fi

# 2. build 3 geometry-filtered arms (baseline_all / geom_top50 / random_top50)
"$PY" "$MODULE_DIR/build_geometry_filtered_dataset.py" \
  --input "$HEAD" --outdir "$DS_DIR" --keep_frac 0.5 --seed 42

# 3. train Qwen2.5-14B student on each arm (Stage-1 only), eval on 991-test
STAGE1="$SHARED_DIR/train_choice_head_distill.py"
for arm in baseline_all geom_top50 random_top50; do
  out_dir="$RUN_ROOT/${arm}_seed${SEED}/stage1_head"
  mkdir -p "$out_dir"
  echo "=== [Phase2 $LABEL] arm=$arm seed=$SEED ==="
  "$PY" "$STAGE1" \
    --model_name "$BASE_MODEL_14B" \
    --data_path "$DS_DIR/train_${arm}.jsonl" \
    --val_path "$DATA15/val.jsonl" \
    --test_path "$DATA15/test.jsonl" \
    --output_dir "$out_dir" \
    --num_epochs 1 --batch_size 2 --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 --rank 16 --lora_alpha 32 --alpha 0.35 \
    --default_distill_mask 0 --seed "$SEED" --deterministic \
    2>&1 | tee "$RUN_ROOT/logs/${arm}_seed${SEED}.log"
done

echo ""
echo "=== Phase2 $LABEL seed=$SEED test accuracy ==="
for arm in baseline_all geom_top50 random_top50; do
  acc=$(grep -aE "test_acc=|测试集准确率" "$RUN_ROOT/logs/${arm}_seed${SEED}.log" | tail -1 || true)
  echo "  $arm : $acc"
done
echo "Run dir: $RUN_ROOT"
