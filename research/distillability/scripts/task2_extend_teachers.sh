#!/usr/bin/env bash
# 任务2: 给 Qwen14B + Qwen32B 补 CMExam 3-seed 蒸馏对照, 把 H1 的 N 从 4 扩到 6。
# 幂等: 已完成的 arm-seed 会跳过(脚本内部 run_phase2 不重跑已存在的 best)。
set -uo pipefail
cd /home/student/arthas/mentalDistill
source setup.env 2>/dev/null
DIST=research/distillability
LOG="$DIST/task2_extend_teachers.log"
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

gpu_busy() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{if($1>2000) print "busy"; else print "free"}'; }

log "=== 任务2启动: 扩展教师 Qwen14B + Qwen32B ==="
for teacher in qwen14b qwen32b; do
  if grep -q "ALL_${teacher}_DONE" "$LOG" 2>/dev/null; then
    log "$teacher 已完成, 跳过"; continue
  fi
  while [ "$(gpu_busy)" = "busy" ]; do log "GPU忙,等待..."; sleep 60; done
  log "--- $teacher 3-seed 蒸馏 ---"
  for s in 42 8 11; do
    log "  $teacher seed=$s"
    SEED=$s bash "$DIST/scripts/run_phase2_distill.sh" "$teacher" >> "$DIST/${teacher}_distill_3seed.log" 2>&1
  done
  log "ALL_${teacher}_DONE"
done

log "=== 任务2完成, 重算 H1 (N=6) ==="
"$EASYEDIT_PY" "$DIST/scripts/h1_baseline_comparison.py" > "$DIST/outputs/h1_comparison_N6_run.log" 2>&1 || log "WARN h1重算失败"
log "=== 任务2全部完成 ==="
echo "TASK2_DONE" >> "$LOG"
