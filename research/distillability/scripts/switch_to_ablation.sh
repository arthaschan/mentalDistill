#!/usr/bin/env bash
set -uo pipefail
# 切换编排器: 等当前 Phi4 seed8 跑完 -> 暂停项目流水线 -> 跑 α 消融21点 -> 跑完自动恢复项目流水线。
# 设计为 nohup 后台运行, VSCode/SSH 断开也会继续。
cd /home/student/arthas/mentalDistill
source setup.env 2>/dev/null
PY="${EASYEDIT_PY:-/home/student/anaconda3/bin/python3}"
DIST=research/distillability
LOG="$DIST/switch_to_ablation.log"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
gpu_used(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1; }

log "=== 切换编排器启动: 等当前 Phi4 seed8 训练完成 ==="
# 步骤1: 等当前 run (Phi4 seed8) 干净结束
while ! grep -q "MEDQA_Phi4_seed8_DONE" "$DIST/task3_medqa.log" 2>/dev/null; do
  sleep 60
done
log "Phi4 seed8 完成, 暂停项目流水线"

# 步骤2: 停掉任务3 relay + 其训练子进程, 防止它启动 seed11 抢卡。
# 此刻唯一的 train_choice 进程是任务3的, 消融尚未启动, pkill 安全。
pkill -f "task3_medqa_relay.sh"        2>/dev/null && log "已停 task3 relay"
sleep 2
pkill -f "run_medqa_distill.sh"        2>/dev/null && log "已停 run_medqa_distill"
sleep 2
pkill -f "train_choice_head_distill.py" 2>/dev/null && log "已停 训练进程"
sleep 10

# 步骤3: 等 GPU 真正释放(<2GB)再启动消融, 杜绝 OOM
log "等 GPU 释放..."
while [ "$(gpu_used)" -gt 2000 ]; do sleep 20; done
log "GPU 已空闲($(gpu_used)MB), 启动 α 消融 21 点"

# 步骤4: 前台跑消融(21点 7α×3seed, 幂等), 跑完汇总
bash 15_fulldata_resplit/scripts/run_alpha_ablation.sh \
  > 15_fulldata_resplit/logs/alpha_ablation_main.log 2>&1
log "消融训练结束, 跑汇总+选点"
"$PY" 15_fulldata_resplit/scripts/summarize_alpha_ablation.py \
  > 15_fulldata_resplit/logs/alpha_summary.log 2>&1 || log "WARN 汇总脚本失败(可手动跑)"
log "=== α 消融全部完成 ==="
echo "ABLATION_DONE" >> "$LOG"

# 步骤5: 恢复项目流水线 —— 重启任务3 relay(幂等, 自动跳过 Phi4 s42/s8, 从 s11 续)
# 后续 泛化/D2/P3 进程仍在后台等各自标记, 任务3完成写 TASK3_DONE 后会自动接力。
log "恢复项目流水线: 重启 task3 relay (幂等续跑)"
nohup bash "$DIST/scripts/task3_medqa_relay.sh" > "$DIST/task3_resumed_stdout.log" 2>&1 &
log "项目流水线已恢复 (PID $!). 切换编排全部完成。"
echo "SWITCH_ALL_DONE" >> "$LOG"
