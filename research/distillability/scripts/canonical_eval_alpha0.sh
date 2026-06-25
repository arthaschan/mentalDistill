#!/usr/bin/env bash
set -uo pipefail
# canonical eval 插队版: 等当前 Yi34B seed42 这个 run 干净结束 -> 暂停任务3 ->
# 立刻插队跑 α=0 的 canonical eval(full991+dental125, 3种子) -> 跑完恢复任务3 relay。
# nohup/setsid 守护, 断线继续。幂等。
cd /home/student/arthas/mentalDistill
source setup.env 2>/dev/null
PY="${EASYEDIT_PY:-/home/student/anaconda3/bin/python3}"
DIST=research/distillability
ABL=15_fulldata_resplit/runs/alpha_ablation_14b
LOG="$DIST/canonical_eval_alpha0.log"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
gpu_used(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null|head -1; }

if grep -q "CANON_EVAL_DONE" "$LOG" 2>/dev/null; then log "已完成,跳过"; exit 0; fi

# 步骤1: 等当前 Yi34B seed42 run 干净结束(写出 _DONE 标记)
log "=== canonical插队: 等当前 Yi34B seed42 完成 ==="
while ! grep -q "MEDQA_Yi34B_seed42_DONE" "$DIST/task3_medqa.log" 2>/dev/null; do sleep 30; done
log "Yi34B seed42 完成, 暂停任务3 relay 插队"

# 步骤2: 停任务3 relay + 其训练子进程(此刻无其它训练, 安全)
pkill -f "task3_medqa_relay.sh"        2>/dev/null && log "已停 task3 relay"
sleep 2
pkill -f "run_medqa_distill.sh"        2>/dev/null && log "已停 run_medqa_distill"
sleep 2
pkill -f "train_choice_head_distill.py" 2>/dev/null && log "已停 训练进程"
sleep 8
while [ "$(gpu_used)" -gt 2000 ]; do log "等GPU释放($(gpu_used)MB)..."; sleep 15; done
log "GPU空闲($(gpu_used)MB), 开始 α=0 canonical eval"

# 步骤3: 评估 α=0 的 3 种子 (full + dental)
BASE="${BASE_MODEL_14B:-$PWD/models/Qwen2.5-14B-Instruct}"
EVAL="shared/evaluate_model.py"
FULL=15_fulldata_resplit/data/test.jsonl
DENTAL=15_fulldata_resplit/data/test_dental.jsonl
for seed in 11 42 8; do
  adir="$ABL/outputs/a0p0_s${seed}/stage1_head/best"
  [ -f "$adir/adapter_config.json" ] || { log "WARN 缺 $adir,跳过"; continue; }
  for tname in full dental; do
    tfile=$([ "$tname" = full ] && echo "$FULL" || echo "$DENTAL")
    marker="CANON_a0_s${seed}_${tname}_DONE"
    grep -q "$marker" "$LOG" 2>/dev/null && { log "$marker 已存在,跳过"; continue; }
    log "评估 α=0 seed=$seed [$tname]"
    "$PY" "$EVAL" --base_model "$BASE" --adapter_dir "$adir" --test_data "$tfile" \
      --wrong_log "$ABL/canon_wrong_a0_s${seed}_${tname}.jsonl" \
      >> "$DIST/canon_eval_a0_s${seed}_${tname}.log" 2>&1 \
      && log "$marker" || log "WARN α=0 s$seed $tname 失败"
  done
done

# 步骤4: 汇总
log "=== canonical eval 完成, 汇总 ==="
"$PY" - <<'PYEOF' 2>&1 | tee -a "$LOG"
import re,os
base="research/distillability"
def acc(f):
    if not os.path.exists(f): return None
    cands=[]
    for line in open(f,encoding="utf-8",errors="ignore"):
        if re.search(r'accuracy|正确率|准确率|correct',line,re.I):
            m=re.search(r'(\d+\.?\d*)\s*%',line) or re.search(r'(\d+)\s*/\s*(\d+)',line) or re.search(r'[:：]\s*(\d+\.\d+)',line)
            if m:
                if m.lastindex==2: cands.append(100*int(m.group(1))/int(m.group(2)))
                else: cands.append(float(m.group(1)))
    return cands[-1] if cands else None
for tname in ["full","dental"]:
    vals=[]
    for s in [11,42,8]:
        a=acc(f"{base}/canon_eval_a0_s{s}_{tname}.log")
        if a is not None: vals.append(a)
        print(f"  α=0 seed={s} [{tname}]: {a}")
    if vals: print(f"  >>> α=0 [{tname}] canonical 均值 = {sum(vals)/len(vals):.2f}% (n={len(vals)})")
print("对比: 教师87.18% / 14B零样本83.55% / 论文主结果(α=0.35)88.67%")
PYEOF
log "CANON_EVAL_DONE"; echo "CANON_EVAL_DONE" >> "$LOG"

# 步骤5: 恢复任务3 relay(幂等, 自动跳过已完成, 从 Yi34B seed8 续)
log "恢复任务3 relay (幂等续跑)"
nohup bash "$DIST/scripts/task3_medqa_relay.sh" > "$DIST/task3_resumed2_stdout.log" 2>&1 &
log "任务3已恢复(PID $!). canonical插队全部完成。"
echo "CANON_PRIORITY_ALL_DONE" >> "$LOG"
