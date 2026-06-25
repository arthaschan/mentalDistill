#!/usr/bin/env bash
set -euo pipefail
# ============================================================
# Choice-Head KL 蒸馏 — α 消融实验编排脚本 (7B, Stage-1 only)
# Module 21_ablation
#
# 与 14B 消融 (run_alpha_ablation_14b.sh) 完全平行, 唯一区别:
#   - 学生模型: Qwen2.5-7B-Instruct
#   - 学习率:   1.2e-4 (对齐 7B 主实验 grid_params_7b.json, 注意≠14B的1e-4)
# 唯一实验变量仍是 --alpha (KL 权重). 数据/种子/其它超参全部冻结.
#
# 目的: 验证"最优 α 是否随学生容量变化"——理论预期弱学生(7B)
#       可能更需要软标签正则, 最优 α 或右移 (>0).
#
# 幂等: 已存在 best/adapter_config.json 的 (alpha,seed) 自动跳过.
# 顺序: 单 H100, GPU 任务串行 (不并行, 防 OOM).
# ============================================================

# ROOT_DIR = 21_ablation/, PROJECT_ROOT = 仓库根
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$ROOT_DIR/.." && pwd)"
SHARED_DIR="$PROJECT_ROOT/shared"
STAGE1_SCRIPT="$SHARED_DIR/train_choice_head_distill.py"

# Python 解析: 优先 setup.env 的 EASYEDIT_PY, 否则 python3
PY="${EASYEDIT_PY:-python3}"

# 模型路径 (7B 学生)
BASE_MODEL_7B="${BASE_MODEL_7B:-$PROJECT_ROOT/models/Qwen2.5-7B-Instruct}"

# 数据 (21_ablation/data 软链到 module 15 同一份)
DATA_DIR="$ROOT_DIR/data"
TRAIN_HEAD="$DATA_DIR/train_head_distill.jsonl"
VAL_DATA="$DATA_DIR/val.jsonl"
TEST_DATA="$DATA_DIR/test.jsonl"

# 消融输出根目录 (21_ablation/runs, 固定名便于幂等续跑)
RUN_ROOT="$ROOT_DIR/runs/alpha_ablation_7b"
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/outputs"

# 冻结超参 (照搬 grid_params_7b.json — 注意 LR=1.2e-4, 与 14B 的 1e-4 不同)
LR="1.2e-4"; RANK="16"; LORA_ALPHA="32"; BS="2"; ACCUM="4"; EPOCHS="1"

# α 网格 与 种子 (与 14B 消融完全对齐, 便于跨容量比较)
ALPHAS=(0.0 0.15 0.25 0.35 0.50 0.65 1.0)
SEEDS=(11 42 8)

REPORT="$RUN_ROOT/ROLLING_REPORT.md"

echo "==============================================="
echo " α-Ablation (7B, Stage-1 only) — Module 21"
echo " alphas : ${ALPHAS[*]}"
echo " seeds  : ${SEEDS[*]}"
echo " lr     : $LR  (对齐 7B 主实验, ≠14B)"
echo " model  : $BASE_MODEL_7B"
echo " out    : $RUN_ROOT"
echo "==============================================="

# rolling 报告表头 (仅首次创建)
if [[ ! -f "$REPORT" ]]; then
  {
    echo "# α-Ablation Rolling Report (7B, Stage-1 only) — Module 21"
    echo ""
    echo "唯一变量 α (KL 权重). Loss = α·KL + (1−α)·CE. 其它超参冻结同 7B 主实验 (LR=1.2e-4)."
    echo "本表 acc 来自训练内置 evaluate_generation (内部可比). 头条对齐需另跑 canonical eval."
    echo ""
    echo "| α | seed | val_acc(%) | test_acc(builtin,%) | status | finished_at |"
    echo "|---|------|-----------|---------------------|--------|-------------|"
  } > "$REPORT"
fi

alpha_tag() { echo "$1" | sed 's/\./p/'; }   # 0.35 -> 0p35 (目录名安全)

for alpha in "${ALPHAS[@]}"; do
  atag="$(alpha_tag "$alpha")"
  for seed in "${SEEDS[@]}"; do
    name="a${atag}_s${seed}"
    out_dir="$RUN_ROOT/outputs/$name/stage1_head"
    best_cfg="$out_dir/best/adapter_config.json"
    log_path="$RUN_ROOT/logs/stage1_${name}.log"

    if [[ -f "$best_cfg" ]]; then
      echo "[SKIP] $name (already trained)"
      continue
    fi

    mkdir -p "$out_dir"
    echo "[RUN] $name  alpha=$alpha seed=$seed  $(date '+%F %T')"

    set +e
    "$PY" "$STAGE1_SCRIPT" \
      --model_name "$BASE_MODEL_7B" \
      --data_path  "$TRAIN_HEAD" \
      --val_path   "$VAL_DATA" \
      --test_path  "$TEST_DATA" \
      --output_dir "$out_dir" \
      --num_epochs "$EPOCHS" \
      --batch_size "$BS" \
      --gradient_accumulation_steps "$ACCUM" \
      --learning_rate "$LR" \
      --rank "$RANK" \
      --lora_alpha "$LORA_ALPHA" \
      --alpha "$alpha" \
      --default_distill_mask 0 \
      --seed "$seed" \
      --deterministic \
      > "$log_path" 2>&1
    rc=$?
    set -e

    # 从日志抓 val / builtin-test 准确率
    val_acc="$(grep -oE '\[VAL\] epoch=[0-9]+ acc=[0-9.]+' "$log_path" | tail -1 | grep -oE '[0-9.]+$' || echo NA)"
    test_acc="$(grep -oE '\[TEST-BEST\] epoch=[0-9]+ test_acc=[0-9.]+' "$log_path" | tail -1 | grep -oE '[0-9.]+$' || echo NA)"

    if [[ $rc -eq 0 ]]; then
      status="OK"
      echo "[DONE] $name  val=$val_acc test=$test_acc"
    else
      status="FAIL(rc=$rc)"
      echo "[FAIL] $name rc=$rc — see $log_path"
    fi
    echo "| $alpha | $seed | $val_acc | $test_acc | $status | $(date '+%F %T') |" >> "$REPORT"
  done
done

echo ""
echo "==============================================="
echo " α-Ablation (7B) finished. Rolling report: $REPORT"
echo " Next: canonical eval (见 run_canonical_eval.sh)"
echo "==============================================="
