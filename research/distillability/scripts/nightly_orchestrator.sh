#!/usr/bin/env bash
# 夜间自动接力编排器 (idempotent — 可重复运行, 已完成的步骤会跳过)
# 流程: 等Yi蒸馏 -> 启动Gemma蒸馏 -> 跑E1指标家族对比 -> 新数据集teacher logprobs(E2)
# 单 H100, 全部 GPU 任务顺序执行, 避免 OOM。
set -uo pipefail

cd /home/student/arthas/mentalDistill
source setup.env 2>/dev/null
PY="${EASYEDIT_PY:-/home/student/anaconda3/bin/python3}"
DIST=research/distillability
LOG="$DIST/nightly_orchestrator.log"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

# ---------- 工具: 判断某教师的 phase2 是否3seed齐全 ----------
seeds_done() {  # $1=label(小写)
  local label="$1" n=0
  for s in 42 8 11; do
    for arm in baseline_all geom_top50 random_top50; do
      grep -q "\[TEST-BEST\]" "$DIST/runs/phase2_${label}/logs/${arm}_seed${s}.log" 2>/dev/null && n=$((n+1))
    done
  done
  echo "$n"  # 9 = 全齐
}

gpu_busy() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{if($1>2000) print "busy"; else print "free"}'; }

log "=== 夜间编排器启动 ==="

# ---------- 步骤1: 等 Yi 蒸馏完成 ----------
log "步骤1: 等待 Yi 蒸馏 (需9个arm-seed)"
while true; do
  yi=$(seeds_done yi34b)
  # 也兼容 run_phase2 写在 phase2_yi34b 还是其他位置
  if grep -q "ALL_YI_SEEDS_DONE" "$DIST/yi34b_distill_3seed.log" 2>/dev/null; then
    log "  Yi 蒸馏脚本报告完成 (ALL_YI_SEEDS_DONE)"; break
  fi
  if [ "$yi" -ge 9 ]; then
    log "  Yi 9/9 arm-seed 齐全 (双保险跳出)"; break
  fi
  log "  Yi 进度: ${yi}/9 arm-seed, 等待中..."
  sleep 120
done

# ---------- 步骤2: 启动 Gemma 蒸馏 (GPU空闲后) ----------
if grep -q "ALL_GEMMA_SEEDS_DONE" "$DIST/gemma27b_distill_3seed.log" 2>/dev/null; then
  log "步骤2: Gemma 蒸馏已完成, 跳过"
else
  log "步骤2: 等 GPU 空闲后启动 Gemma 3-seed 蒸馏"
  while [ "$(gpu_busy)" = "busy" ]; do log "  GPU 忙, 等待..."; sleep 60; done
  log "  GPU 空闲, 启动 Gemma 蒸馏"
  ( for s in 42 8 11; do
      echo "########## GEMMA SEED $s ##########"
      SEED=$s bash "$DIST/scripts/run_phase2_distill.sh" gemma27b
    done
    echo "ALL_GEMMA_SEEDS_DONE"
  ) > "$DIST/gemma27b_distill_3seed.log" 2>&1
  log "  Gemma 蒸馏完成"
fi

# ---------- 步骤3: E1 指标家族对比 ----------
log "步骤3: 跑 E1 指标家族对比 (5指标 vs 真实增益)"
"$PY" "$DIST/scripts/transferability_scores.py" > "$DIST/outputs/transferability_run.log" 2>&1 || log "  WARN transferability 失败"
"$PY" "$DIST/scripts/h1_baseline_comparison.py" > "$DIST/outputs/h1_comparison_run.log" 2>&1 || log "  WARN h1 对比失败"
log "  E1 完成, 结果见 outputs/h1_comparison_run.log + h1_baseline_comparison.json"

# ---------- 步骤4: 新数据集 teacher logprobs (E2, 便宜的预测一致性) ----------
# 只在已下载好的数据集上, 对一部分代表教师生成 logprobs。GPU顺序执行。
log "步骤4: 新数据集 teacher logprobs (E2)"
# 代表教师 (覆盖强中弱): Qwen32B(强) Yi34B(中) Phi4(弱) — 已在本地有模型
declare -A MODELS=(
  [Qwen32B]=models/Qwen2.5-32B-Instruct
  [Yi34B]=models/Yi-1.5-34B-Chat
  [Phi4]=models/phi-4
)
for ds in mmlu_med mmlu_full medqa; do
  data="data_ext/$ds/test.jsonl"
  [ -f "$data" ] || { log "  [skip] $ds 无 test.jsonl"; continue; }
  for tlabel in Qwen32B Yi34B Phi4; do
    out="$DIST/teacher_labels_ext/${ds}_${tlabel}_logprobs.jsonl"
    if [ -f "$out" ] && [ "$(wc -l < "$out")" -gt 50 ]; then
      log "  [skip] $ds/$tlabel 已存在"; continue
    fi
    while [ "$(gpu_busy)" = "busy" ]; do sleep 30; done
    log "  生成 $ds / $tlabel logprobs"
    mkdir -p "$DIST/teacher_labels_ext"
    "$PY" shared/generate_teacher_labels_local_logprobs.py \
      --model_path "${MODELS[$tlabel]}" \
      --dataset "$data" \
      --output "$out" \
      --gt_field Answer --resume \
      > "$DIST/teacher_labels_ext/${ds}_${tlabel}_gen.log" 2>&1 || log "    WARN $ds/$tlabel 失败"
  done
done
log "  E2 logprobs 完成"

# ---------- 步骤5: E2 跨数据集指标一致性分析 ----------
log "步骤5: E2 跨数据集指标分析"
"$PY" "$DIST/scripts/e2_cross_dataset.py" > "$DIST/outputs/e2_run.log" 2>&1 || log "  WARN E2分析失败"
log "  E2 分析完成, 结果见 outputs/e2_run.log + e2_cross_dataset.json"

log "=== 夜间编排器全部完成 ==="
echo "NIGHTLY_ORCHESTRATOR_DONE" >> "$LOG"
