#!/usr/bin/env bash
set -euo pipefail
# ============================================================
# Choice-Head KL 蒸馏 — α 消融实验编排脚本 (14B, Stage-1 only)
#
# 唯一变量: --alpha (KL 权重). 其它超参/种子/数据/脚本全部冻结,
# 与 module 15 主实验 (configs/grid_params_14b.json) 完全一致.
#
# 幂等: 已存在 best/adapter_config.json 的 (alpha,seed) 自动跳过,
#       可随时 Ctrl-C / 断线后重跑续上.
# 顺序: 单 H100, GPU 任务串行 (不并行, 防 OOM).
# ============================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$ROOT_DIR/.." && pwd)"
SHARED_DIR="$PROJECT_ROOT/shared"
STAGE1_SCRIPT="$SHARED_DIR/train_choice_head_distill.py"

# Python 解析: 优先 setup.env 的 EASYEDIT_PY, 否则 python3
PY="${EASYEDIT_PY:-python3}"

# 模型路径
BASE_MODEL_14B="${BASE_MODEL_14B:-$PROJECT_ROOT/models/Qwen2.5-14B-Instruct}"

# 数据 (与主实验同一份)
DATA_DIR="$ROOT_DIR/data"
TRAIN_HEAD="$DATA_DIR/train_head_distill.jsonl"
VAL_DATA="$DATA_DIR/val.jsonl"
TEST_DATA="$DATA_DIR/test.jsonl"

# 消融输出根目录 (固定名, 便于幂等续跑; 不带时间戳)
RUN_ROOT="$ROOT_DIR/runs/alpha_ablation_14b"
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/outputs"

# 冻结超参 (照搬 grid_params_14b.json)
LR="1e-4"; RANK="16"; LORA_ALPHA="32"; BS="2"; ACCUM="4"; EPOCHS="1"

# α 网格 与 种子
ALPHAS=(0.0 0.15 0.25 0.35 0.50 0.65 1.0)
SEEDS=(11 42 8)

REPORT="$RUN_ROOT/ROLLING_REPORT.md"

echo "==============================================="
echo " α-Ablation (14B, Stage-1 only)"
echo " alphas : ${ALPHAS[*]}"
echo " seeds  : ${SEEDS[*]}"
echo " model  : $BASE_MODEL_14B"
echo " out    : $RUN_ROOT"
echo "==============================================="

# rolling 报告表头 (仅首次创建)
if [[ ! -f "$REPORT" ]]; then
  {
    echo "# α-Ablation Rolling Report (14B, Stage-1 only)"
    echo ""
    echo "唯一变量 α (KL 权重). Loss = α·KL + (1−α)·CE. 其它超参冻结同 module 15 主实验."
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
      --model_name "$BASE_MODEL_14B" \
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
echo " α-Ablation finished. Rolling report: $REPORT"
echo " Next: python3 $ROOT_DIR/scripts/summarize_alpha_ablation.py"
echo "==============================================="
