#!/usr/bin/env bash
set -uo pipefail
# ============================================================
# 7B α-消融 插队编排 (仿 canonical_eval_alpha0.sh 的暂停/恢复范式)
#
# 流程:
#   1. 等当前 Yi34B seed8 run 干净结束 (写出 MEDQA_Yi34B_seed8_DONE)
#   2. 暂停任务3 relay + 训练子进程, 等 GPU 释放
#   3. 跑 7B α-消融 21 点 (run_alpha_ablation_7b.sh, 幂等)
#   4. 对 7B 全部 (α,seed) 做 canonical eval (用已验证的 evaluate_model.py 直评,
#      不用有 bug 的 run_eval_dual.py), full991 + dental125
#   5. 汇总均值 -> 写日志
#   6. 恢复任务3 relay (幂等, 从 Yi34B s11 续)
#
# setsid/nohup 守护, 断线继续。幂等可重跑。
# ============================================================
cd /home/student/arthas/mentalDistill
source setup.env 2>/dev/null
PY="${EASYEDIT_PY:-/home/student/anaconda3/bin/python3}"
DIST=research/distillability
ABL=15_fulldata_resplit/runs/alpha_ablation_7b
LOG="$DIST/ablation_7b_priority.log"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
gpu_used(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null|head -1; }

if grep -q "ABLATION_7B_ALL_DONE" "$LOG" 2>/dev/null; then log "已全部完成,跳过"; exit 0; fi

# ---- 步骤1: 等当前 Yi34B seed8 run 干净结束 ----
log "=== 7B消融插队: 等当前 Yi34B seed8 完成 ==="
while ! grep -q "MEDQA_Yi34B_seed8_DONE" "$DIST/task3_medqa.log" 2>/dev/null; do sleep 30; done
log "Yi34B seed8 完成, 暂停任务3 relay 插队"

# ---- 步骤2: 停任务3 relay + 训练子进程, 等 GPU 释放 ----
pkill -f "task3_medqa_relay.sh"          2>/dev/null && log "已停 task3 relay"
sleep 2
pkill -f "run_medqa_distill.sh"          2>/dev/null && log "已停 run_medqa_distill"
sleep 2
pkill -f "train_choice_head_distill.py"  2>/dev/null && log "已停 训练进程"
sleep 8
while [ "$(gpu_used)" -gt 2000 ]; do log "等GPU释放($(gpu_used)MB)..."; sleep 15; done
log "GPU空闲($(gpu_used)MB), 开始 7B α-消融 21 点"

# ---- 步骤3: 跑 7B α-消融 (幂等, 已完成的点自动跳过) ----
bash 15_fulldata_resplit/scripts/run_alpha_ablation_7b.sh >> "$DIST/ablation_7b_train_stdout.log" 2>&1 \
  && log "7B α-消融 21 点训练完成" || log "WARN 7B α-消融训练非零退出 (检查 ablation_7b_train_stdout.log)"

# ---- 步骤4: 7B canonical eval (直评 evaluate_model.py, full+dental, 全 α×seed) ----
log "=== 7B canonical eval (full991 + dental125) ==="
BASE="${BASE_MODEL_7B:-$PWD/models/Qwen2.5-7B-Instruct}"
EVAL="shared/evaluate_model.py"
FULL=15_fulldata_resplit/data/test.jsonl
DENTAL=15_fulldata_resplit/data/test_dental.jsonl
ALPHAS=(0p0 0p15 0p25 0p35 0p50 0p65 1p0)
SEEDS=(11 42 8)
for atag in "${ALPHAS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    adir="$ABL/outputs/a${atag}_s${seed}/stage1_head/best"
    [ -f "$adir/adapter_config.json" ] || { log "WARN 缺 $adir,跳过"; continue; }
    for tname in full dental; do
      tfile=$([ "$tname" = full ] && echo "$FULL" || echo "$DENTAL")
      marker="CANON7B_a${atag}_s${seed}_${tname}_DONE"
      grep -q "$marker" "$LOG" 2>/dev/null && { log "$marker 已存在,跳过"; continue; }
      log "评估 7B α=$atag seed=$seed [$tname]"
      "$PY" "$EVAL" --base_model "$BASE" --adapter_dir "$adir" --test_data "$tfile" \
        --wrong_log "$ABL/canon_wrong_a${atag}_s${seed}_${tname}.jsonl" \
        >> "$DIST/canon7b_a${atag}_s${seed}_${tname}.log" 2>&1 \
        && log "$marker" || log "WARN 7B α=$atag s$seed $tname 失败"
    done
  done
done

# ---- 步骤5: 汇总均值 ----
log "=== 7B canonical eval 完成, 汇总 ==="
"$PY" - <<'PYEOF' 2>&1 | tee -a "$LOG"
import re,os
base="research/distillability"
ALPHAS=["0p0","0p15","0p25","0p35","0p50","0p65","1p0"]
ALABEL={"0p0":"0.0","0p15":"0.15","0p25":"0.25","0p35":"0.35","0p50":"0.50","0p65":"0.65","1p0":"1.0"}
SEEDS=[11,42,8]
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
print("\n=== 7B α-消融 canonical eval 汇总 ===")
for tname in ["full","dental"]:
    print(f"\n[{tname}]")
    for atag in ALPHAS:
        vals=[]
        for s in SEEDS:
            a=acc(f"{base}/canon7b_a{atag}_s{s}_{tname}.log")
            if a is not None: vals.append(a)
        if vals:
            mean=sum(vals)/len(vals)
            print(f"  α={ALABEL[atag]:<5} mean={mean:.2f}%  n={len(vals)}  seeds={['%.2f'%v for v in vals]}")
        else:
            print(f"  α={ALABEL[atag]:<5} (无结果)")
print("\n对比锚点: 教师87.18% / 论文14B主结果(α=0.35)88.67% / 14B消融α=0 canonical full=89.14%")
PYEOF
log "ABLATION_7B_ALL_DONE"; echo "ABLATION_7B_ALL_DONE" >> "$LOG"

# ---- 步骤6: 恢复任务3 relay (幂等, 从 Yi34B s11 续) ----
log "恢复任务3 relay (幂等续跑)"
nohup bash "$DIST/scripts/task3_medqa_relay.sh" > "$DIST/task3_resumed_after7b_stdout.log" 2>&1 &
log "任务3已恢复(PID $!). 7B 消融插队全部完成。"
echo "ABLATION_7B_PRIORITY_ALL_DONE" >> "$LOG"
