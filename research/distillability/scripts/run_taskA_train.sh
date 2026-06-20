#!/usr/bin/env bash
set -euo pipefail
# Task A: geometry-filtered distillation — 3 arms, seed-matched.
# Trains Qwen2.5-14B student on Llama70B teacher labels under 3 KL-supervision sets:
#   baseline_all  : all 2223 clean_teacher samples get KL
#   geom_top50    : top-50% by geometry distillability score get KL
#   random_top50  : random 50% get KL (control)
# All arms identical except WHICH samples carry teacher KL. Stage-1 only.

# Script lives at research/distillability/scripts/run_taskA_train.sh
MODULE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # research/distillability
REPO_ROOT="$(cd "$MODULE_DIR/../.." && pwd)"                    # repo root
# common_env.sh expects ROOT_DIR to be one level under repo root (REBUILD_ROOT=ROOT_DIR/..).
# Point ROOT_DIR at research/ so REBUILD_ROOT resolves to the repo root.
ROOT_DIR="$REPO_ROOT/research"
source "$REPO_ROOT/shared/common_env.sh"
resolve_python
resolve_model_dir BASE_MODEL_14B Qwen2.5-14B-Instruct

DS_DIR="${DS_DIR:-$MODULE_DIR/datasets}"
DATA15="$REPO_ROOT/15_fulldata_resplit/data"
RUN_TAG="${RUN_TAG:-taskA}"
RUN_ROOT="$MODULE_DIR/runs/$(date +%Y%m%d_%H%M%S)_${RUN_TAG}"
STAGE1="$REPO_ROOT/shared/train_choice_head_distill.py"
SEED="${SEED:-42}"

mkdir -p "$RUN_ROOT/logs"

for arm in baseline_all geom_top50 random_top50; do
  out_dir="$RUN_ROOT/$arm/stage1_head"
  mkdir -p "$out_dir"
  echo "=== [Task A] arm=$arm seed=$SEED ==="
  "$PY" "$STAGE1" \
    --model_name "$BASE_MODEL_14B" \
    --data_path "$DS_DIR/train_${arm}.jsonl" \
    --val_path "$DATA15/val.jsonl" \
    --test_path "$DATA15/test.jsonl" \
    --output_dir "$out_dir" \
    --num_epochs 1 \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --rank 16 --lora_alpha 32 \
    --alpha 0.35 \
    --default_distill_mask 0 \
    --seed "$SEED" \
    --deterministic \
    2>&1 | tee "$RUN_ROOT/logs/${arm}.log"
done

echo ""
echo "=== Task A done. Test accuracy per arm: ==="
for arm in baseline_all geom_top50 random_top50; do
  acc=$(grep -iE "test_acc=|测试集准确率" "$RUN_ROOT/logs/${arm}.log" | tail -1 || true)
  echo "  $arm : $acc"
done
echo "Run dir: $RUN_ROOT"
