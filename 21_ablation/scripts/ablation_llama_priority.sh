#!/usr/bin/env bash
set -uo pipefail
# ============================================================
# Llama 组合③ α-消融 插队编排 (Module 21) — 仿 ablation_7b_priority.sh 暂停/恢复范式
#
# 组合③: Llama-3.3-70B 教师 → Qwen2.5-14B 学生, clean_teacher 2223 子集 (不一致率 48.4%)
#
# 流程:
#   1. 等当前 task3 正在跑的 run (Qwen32B seed42) 干净结束 (写出 MEDQA_Qwen32B_seed42_DONE)
#   2. 暂停任务3 relay + 训练子进程, 等 GPU 释放
#   3. 跑 Llama α-消融 21 点 (run_alpha_ablation_llama.sh, 幂等)
#   4. canonical eval (已验证的 evaluate_model.py 直评, full991+dental125)
#   5. 汇总均值 -> 写日志
#   6. 恢复任务3 relay (幂等, 从 Qwen32B 余下 seed 续)
#
# setsid/nohup 守护, 断线继续。幂等可重跑。
# 所有产物落 21_ablation/, 日志落 21_ablation/logs/。
# ============================================================
cd /home/student/arthas/mentalDistill
source setup.env 2>/dev/null
PY="${EASYEDIT_PY:-/home/student/anaconda3/bin/python3}"
DIST=research/distillability                    # 任务3 relay 仍在这里
ABL=21_ablation/runs/alpha_ablation_llama       # Llama 消融产物
LOG=21_ablation/logs/ablation_llama_priority.log
mkdir -p 21_ablation/logs
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
gpu_used(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null|head -1; }

if grep -q "ABLATION_LLAMA_ALL_DONE" "$LOG" 2>/dev/null; then log "已全部完成,跳过"; exit 0; fi

# ---- 步骤1: 等当前 task3 正在跑的 run (Qwen32B seed42) 干净结束 ----
# 若 Qwen32B seed42 已 DONE (task3 已推进更远) 也直接通过, 不阻塞。
log "=== Llama消融插队(Module21): 等当前 Qwen32B seed42 完成 ==="
while ! grep -q "MEDQA_Qwen32B_seed42_DONE\|TASK3_DONE" "$DIST/task3_medqa.log" 2>/dev/null; do sleep 30; done
log "当前 task3 run 完成, 暂停任务3 relay 插队"

# ---- 步骤2: 停任务3 relay + 训练子进程, 等 GPU 释放 ----
pkill -f "task3_medqa_relay.sh"          2>/dev/null && log "已停 task3 relay"
sleep 2
pkill -f "run_medqa_distill.sh"          2>/dev/null && log "已停 run_medqa_distill"
sleep 2
pkill -f "train_choice_head_distill.py"  2>/dev/null && log "已停 训练进程"
sleep 8
while [ "$(gpu_used)" -gt 2000 ]; do log "等GPU释放($(gpu_used)MB)..."; sleep 15; done
log "GPU空闲($(gpu_used)MB), 开始 Llama α-消融 21 点"

# ---- 步骤3: 跑 Llama α-消融 (幂等, 已完成的点自动跳过) ----
bash 21_ablation/scripts/run_alpha_ablation_llama.sh >> 21_ablation/logs/ablation_llama_train_stdout.log 2>&1 \
  && log "Llama α-消融 21 点训练完成" || log "WARN Llama α-消融训练非零退出 (检查 logs/ablation_llama_train_stdout.log)"

# ---- 步骤4: canonical eval (直评 evaluate_model.py, full+dental, 全 α×seed) ----
log "=== Llama canonical eval (full991 + dental125) ==="
BASE="${BASE_MODEL_14B:-$PWD/models/Qwen2.5-14B-Instruct}"
EVAL="shared/evaluate_model.py"
FULL=21_ablation/data/test.jsonl
DENTAL=21_ablation/data/test_dental.jsonl
ALPHAS=(0p0 0p15 0p25 0p35 0p50 0p65 1p0)
SEEDS=(11 42 8)
for atag in "${ALPHAS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    adir="$ABL/outputs/a${atag}_s${seed}/stage1_head/best"
    [ -f "$adir/adapter_config.json" ] || { log "WARN 缺 $adir,跳过"; continue; }
    for tname in full dental; do
      tfile=$([ "$tname" = full ] && echo "$FULL" || echo "$DENTAL")
      marker="CANONLLAMA_a${atag}_s${seed}_${tname}_DONE"
      grep -q "$marker" "$LOG" 2>/dev/null && { log "$marker 已存在,跳过"; continue; }
      log "评估 Llama α=$atag seed=$seed [$tname]"
      "$PY" "$EVAL" --base_model "$BASE" --adapter_dir "$adir" --test_data "$tfile" \
        --wrong_log "$ABL/canon_wrong_a${atag}_s${seed}_${tname}.jsonl" \
        >> "21_ablation/logs/canonllama_a${atag}_s${seed}_${tname}.log" 2>&1 \
        && log "$marker" || log "WARN Llama α=$atag s$seed $tname 失败"
    done
  done
done

# ---- 步骤5: 汇总均值 ----
log "=== Llama canonical eval 完成, 汇总 ==="
"$PY" - <<'PYEOF' 2>&1 | tee -a "$LOG"
import re,os
base="21_ablation/logs"
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
print("\n=== Llama 组合③ α-消融 canonical eval 汇总 (Module 21) ===")
for tname in ["full","dental"]:
    print(f"\n[{tname}]")
    for atag in ALPHAS:
        vals=[]
        for s in SEEDS:
            a=acc(f"{base}/canonllama_a{atag}_s{s}_{tname}.log")
            if a is not None: vals.append(a)
        if vals:
            mean=sum(vals)/len(vals)
            print(f"  α={ALABEL[atag]:<5} mean={mean:.2f}%  n={len(vals)}  seeds={['%.2f'%v for v in vals]}")
        else:
            print(f"  α={ALABEL[atag]:<5} (无结果)")
print("\n对比锚点: 教师(Llama70B,不一致48.4%) / 14B消融α=0 canonical full=89.14% / 7B消融α=0.15 full=85.87%")
print("预测验证: 教师越差→KL越有害→α=0更优、α=1.0崩塌更severe (对照 14B 组合①)")
PYEOF
log "ABLATION_LLAMA_ALL_DONE"; echo "ABLATION_LLAMA_ALL_DONE" >> "$LOG"

# ---- 步骤6: 恢复任务3 relay (幂等, 从 Qwen32B 余下 seed 续) ----
log "恢复任务3 relay (幂等续跑)"
nohup bash "$DIST/scripts/task3_medqa_relay.sh" > "$DIST/task3_resumed_after_llama_stdout.log" 2>&1 &
log "任务3已恢复(PID $!). Llama 消融插队全部完成。"
echo "ABLATION_LLAMA_PRIORITY_ALL_DONE" >> "$LOG"
