#!/usr/bin/env bash
set -uo pipefail
# 任务3 自动接力编排器: 等任务2(扩教师)跑完 -> MedQA 跨数据集蒸馏验证。
# 对 3 个代表教师(强/中/弱)做完整 3-seed 三臂蒸馏, 验证"几何去噪增益跨数据集是否成立"。
# 幂等: 已完成的 arm-seed 跳过。单 H100, 顺序执行。
cd /home/student/arthas/mentalDistill
source setup.env 2>/dev/null
DIST=research/distillability
LOG="$DIST/task3_medqa.log"
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
gpu_busy() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{if($1>2000)print"busy";else print"free"}'; }

log "=== 任务3启动: 先等任务2完成 ==="
# 步骤0: 等任务2跑完 (看 TASK2_DONE 标志)
while true; do
  if grep -q "TASK2_DONE" "$DIST/task2_extend_teachers.log" 2>/dev/null; then
    log "任务2已完成, 开始任务3"; break
  fi
  log "等待任务2... (当前GPU: $(gpu_busy))"
  sleep 180
done

# 步骤1: 3个代表教师 (强Qwen32B / 中Yi34B / 弱Phi4), MedQA 完整蒸馏
declare -A MODELS=(
  [Qwen32B]=models/Qwen2.5-32B-Instruct
  [Yi34B]=models/Yi-1.5-34B-Chat
  [Phi4]=models/phi-4
)
for teacher in Phi4 Yi34B Qwen32B; do   # 先弱后强(弱教师是主结果)
  for s in 42 8 11; do
    marker="MEDQA_${teacher}_seed${s}_DONE"
    if grep -q "$marker" "$LOG" 2>/dev/null; then
      log "$teacher seed$s 已完成, 跳过"; continue
    fi
    while [ "$(gpu_busy)" = "busy" ]; do sleep 60; done
    log "--- MedQA $teacher seed=$s ---"
    SEED=$s bash "$DIST/scripts/run_medqa_distill.sh" "$teacher" "${MODELS[$teacher]}" \
      >> "$DIST/medqa_${teacher}_distill.log" 2>&1 && log "$marker" || log "WARN $teacher seed$s 失败"
  done
done

# 步骤2: 汇总 MedQA 三臂结果
log "=== 任务3完成, 汇总 MedQA 结果 ==="
"$EASYEDIT_PY" "$DIST/scripts/summarize_medqa.py" > "$DIST/outputs/medqa_summary.log" 2>&1 || log "WARN 汇总失败(脚本可能未建)"
log "=== 任务3全部完成 ==="
echo "TASK3_DONE" >> "$LOG"
