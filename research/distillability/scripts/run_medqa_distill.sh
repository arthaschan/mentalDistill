#!/usr/bin/env bash
set -uo pipefail
# 任务3: MedQA 跨数据集蒸馏验证 "几何去噪增益是否跨数据集成立"。
# 对每个代表教师: 生成 MedQA train logprobs -> 构建 baseline/geom/random 三臂 -> 3-seed 蒸馏。
# 英文 prompt (DISTILL_PROMPT_LANG=en)。学生 Qwen2.5-14B (同 CMExam, 保证可比)。
#
# 用法: bash research/distillability/scripts/run_medqa_distill.sh <teacher_label> <model_path>
# 例:   bash run_medqa_distill.sh Phi4 models/phi-4

LABEL="${1:?usage: run_medqa_distill.sh <teacher_label> <model_path>}"
MODEL_PATH="${2:?need teacher model path}"
MODULE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$MODULE_DIR/../.." && pwd)"
ROOT_DIR="$REPO_ROOT/research"
source "$REPO_ROOT/shared/common_env.sh"
resolve_python
resolve_model_dir BASE_MODEL_14B Qwen2.5-14B-Instruct

export DISTILL_PROMPT_LANG=en          # MedQA 是英文
DATA="$REPO_ROOT/data_ext/medqa"
LOGPROBS="$MODULE_DIR/teacher_labels_ext/medqa_${LABEL}_train_logprobs.jsonl"
DS_DIR="$MODULE_DIR/datasets_medqa/${LABEL}"
RUN_ROOT="$MODULE_DIR/runs/medqa_${LABEL}"
SEED="${SEED:-42}"
mkdir -p "$DS_DIR" "$RUN_ROOT/logs"

# 1. 生成 MedQA train 集 logprobs (若没有)
if [ ! -f "$LOGPROBS" ] || [ "$(wc -l < "$LOGPROBS")" -lt 100 ]; then
  echo "=== [MedQA $LABEL] 生成 train logprobs ==="
  "$PY" "$REPO_ROOT/shared/generate_teacher_labels_local_logprobs.py" \
    --model_path "$MODEL_PATH" \
    --dataset "$DATA/train.jsonl" \
    --output "$LOGPROBS" \
    --gt_field Answer --resume
fi

# 2. 构建 head-distill 数据集 (从真实 logprobs)
HEAD="$DS_DIR/train_head_distill.jsonl"
if [ ! -f "$HEAD" ]; then
  echo "=== [MedQA $LABEL] build head-distill ==="
  "$PY" "$SHARED_DIR/build_selective_distill_dataset.py" \
    --gt_data "$DATA/train.jsonl" \
    --teacher_soft "$LOGPROBS" \
    --output "$HEAD" \
    --report "$DS_DIR/distill_dataset_report.txt" \
    --min_entropy 0.01 --smooth_eps 0.0 --min_margin 0.0
fi

# 3. 构建 3 臂 (baseline/geom/random)
"$PY" "$MODULE_DIR/build_geometry_filtered_dataset.py" \
  --input "$HEAD" --outdir "$DS_DIR" --keep_frac 0.5 --seed 42

# 4. 三臂 3-seed 蒸馏
STAGE1="$SHARED_DIR/train_choice_head_distill.py"
for arm in baseline_all geom_top50 random_top50; do
  out_dir="$RUN_ROOT/${arm}_seed${SEED}/stage1_head"
  mkdir -p "$out_dir"
  echo "=== [MedQA $LABEL] arm=$arm seed=$SEED ==="
  "$PY" "$STAGE1" \
    --model_name "$BASE_MODEL_14B" \
    --data_path "$DS_DIR/train_${arm}.jsonl" \
    --val_path "$DATA/val.jsonl" \
    --test_path "$DATA/test.jsonl" \
    --output_dir "$out_dir" \
    --num_epochs 1 --batch_size 2 --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 --rank 16 --lora_alpha 32 --alpha 0.35 \
    --default_distill_mask 0 --seed "$SEED" --deterministic \
    2>&1 | tee "$RUN_ROOT/logs/${arm}_seed${SEED}.log"
done

echo "=== MedQA $LABEL seed=$SEED test accuracy ==="
for arm in baseline_all geom_top50 random_top50; do
  acc=$(grep -aE "test_acc=|\[TEST-BEST\]" "$RUN_ROOT/logs/${arm}_seed${SEED}.log" | tail -1 || true)
  echo "  $arm : $acc"
done
echo "Run dir: $RUN_ROOT"
