#!/usr/bin/env bash
set -uo pipefail
# 泛化验证: 用【未参与研究】的 Qwen2.5-7B 跑体检器, 看工具是否对新模型同样有效。
# 等 GPU 空闲(任务2/3 让出)后自动现场生成 logprobs 并体检。幂等。
cd /home/student/arthas/mentalDistill
source setup.env 2>/dev/null
DIST=research/distillability
LOG="$DIST/genvalidation.log"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
gpu_busy(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null|awk '{if($1>20000)print"busy";else print"free"}'; }

MODEL=models/Qwen2.5-7B-Instruct
LABEL=Qwen7B_holdout
DATA=15_fulldata_resplit/data/train.jsonl

if grep -q "GENVAL_DONE" "$LOG" 2>/dev/null; then log "已完成, 跳过"; exit 0; fi

# 依赖门: 等任务3完成(TASK3_DONE)再开始, 避免与 task3 抢GPU
log "=== 泛化验证: 先等任务3完成 ==="
while ! grep -q "TASK3_DONE" "$DIST/task3_medqa.log" 2>/dev/null; do sleep 180; done
log "任务3已完成, 泛化验证开始"

log "=== 泛化验证启动: 等 GPU 空闲后体检 $LABEL (未参与研究的模型) ==="
# 等任务2和任务3都不再大量占显存 (留>20G余量给7B)
while [ "$(gpu_busy)" = "busy" ]; do log "GPU忙(>20G),等待..."; sleep 120; done

log "GPU 空闲, 现场生成 logprobs + 体检"
"$EASYEDIT_PY" "$DIST/scripts/scan_teacher.py" \
  --model_path "$MODEL" --dataset "$DATA" \
  --label "$LABEL" --out_dir "$DIST/reports" \
  >> "$LOG" 2>&1 && log "GENVAL_DONE" || log "WARN 体检失败"

log "=== 泛化验证完成, 报告见 $DIST/reports/${LABEL}_health_report.md ==="
