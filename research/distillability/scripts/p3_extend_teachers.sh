#!/usr/bin/env bash
# P3: 补充 GLM/Gemma/Qwen14B 在 medqa/mmlu_med/mmlu_full 的 teacher logprobs。
# 把3共同教师(已有Qwen32B/Yi34B/Phi4)扩到6教师, 完成 P3 跨域核心表。
# 幂等: 已存在的输出跳过。单H100顺序执行, 带GPU空闲检测避免OOM。
set -uo pipefail
cd /home/student/arthas/mentalDistill
source setup.env 2>/dev/null
PY="${EASYEDIT_PY:-/home/student/anaconda3/bin/python3}"
DIST=research/distillability
LOG="$DIST/p3_extend_teachers.log"
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
gpu_busy() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{if($1>20000) print "busy"; else print "free"}'; }

# 幂等: 已完成则跳过
if grep -q "P3_EXTEND_DONE" "$LOG" 2>/dev/null; then log "P3补教师已完成,跳过"; exit 0; fi

# 依赖门: 等 D2 完成(D2_DONE)再开始, 串行避免抢GPU。D2是当前论文最后一个GPU任务。
log "=== P3补教师: 先等 D2 容量曲线完成 (D2_DONE) ==="
while ! grep -q "D2_DONE" "$DIST/d2_capacity_curve.log" 2>/dev/null; do sleep 180; done
log "D2 已完成, P3 补教师开始"

declare -A MODELS=(
  [GLM32B]=models/GLM-4-32B-0414
  [Gemma27B]=models/gemma-2-27b-it
  [Qwen14B]=models/Qwen2.5-14B-Instruct
)

log "=== P3 补教师 logprobs 启动 (3教师 × 3数据集) ==="
for ds in medqa mmlu_med mmlu_full; do
  data="$DIST/data_ext_rebuilt/$ds/test.jsonl"
  [ -f "$data" ] || { log "[skip] $ds 无重建数据"; continue; }
  for tlabel in GLM32B Gemma27B Qwen14B; do
    out="$DIST/teacher_labels_ext/${ds}_${tlabel}_logprobs.jsonl"
    if [ -f "$out" ] && [ "$(wc -l < "$out")" -gt 50 ]; then
      log "[skip] $ds/$tlabel 已存在 ($(wc -l < "$out")行)"; continue
    fi
    while [ "$(gpu_busy)" = "busy" ]; do log "  GPU忙, 等待P3让行..."; sleep 60; done
    log "生成 $ds / $tlabel logprobs"
    "$PY" shared/generate_teacher_labels_local_logprobs.py \
      --model_path "${MODELS[$tlabel]}" \
      --dataset "$data" \
      --output "$out" \
      --gt_field Answer --resume \
      > "$DIST/teacher_labels_ext/${ds}_${tlabel}_gen.log" 2>&1 \
      && log "  ✅ $ds/$tlabel 完成 ($(wc -l < "$out" 2>/dev/null)行)" \
      || log "  WARN $ds/$tlabel 失败, 见 gen.log"
  done
done

# 6教师齐全后, 重跑跨域分析(自动纳入新教师)
log "补全完成, 重跑 P3 跨域综合分析"
"$PY" "$DIST/scripts/exp_P3_full_crossdomain_6teachers.py" > "$DIST/outputs/p3_6teachers_run.log" 2>&1 \
  && log "  ✅ P3 6教师分析完成, 见 outputs/p3_6teachers_run.log" \
  || log "  WARN P3 6教师分析脚本待创建或失败(可手动跑扩展版)"
log "=== P3 补教师全部完成 ==="
echo "P3_EXTEND_DONE" >> "$LOG"
