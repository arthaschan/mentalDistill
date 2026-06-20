#!/usr/bin/env bash
set -uo pipefail
# D2: 学生容量下限预测 —— 建"学生参数量 vs 蒸馏后正确率"曲线。
# 固定教师(Qwen32B, 最强), 在 CMExam 上蒸馏多个尺寸学生, 看正确率随容量如何变化。
# 用途: 给定目标正确率, 反查能达标的最小学生 = 部署成本下限。
# 排在 任务2/3 + 泛化验证 之后, 等 GPU 空闲自动跑。幂等。
cd /home/student/arthas/mentalDistill
source setup.env 2>/dev/null
DIST=research/distillability
LOG="$DIST/d2_capacity_curve.log"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
gpu_busy(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null|awk '{if($1>20000)print"busy";else print"free"}'; }

if grep -q "D2_DONE" "$LOG" 2>/dev/null; then log "已完成,跳过"; exit 0; fi

# 依赖门: 等泛化验证完成(GENVAL_DONE)再开始, 串行避免抢GPU
log "=== D2: 先等泛化验证完成 ==="
while ! grep -q "GENVAL_DONE" "$DIST/genvalidation.log" 2>/dev/null; do sleep 180; done
log "泛化验证已完成, D2 开始"

# 教师软标签(已有 Qwen32B CMExam logprobs)
TEACHER_LP="$DIST/teacher_labels/qwen32b_train_logprobs.jsonl"
# 用 Qwen32B 教师构建的几何筛选数据集(baseline_all 臂=全量, 用它做干净对照)
DS_DIR="$DIST/datasets/qwen32b"

# 学生尺寸 -> HF repo (小模型先下载)
declare -A REPO=(
  ["0.5B"]="Qwen/Qwen2.5-0.5B-Instruct"
  ["1.5B"]="Qwen/Qwen2.5-1.5B-Instruct"
  ["3B"]="Qwen/Qwen2.5-3B-Instruct"
  ["7B"]="$PWD/models/Qwen2.5-7B-Instruct"
  ["14B"]="$PWD/models/Qwen2.5-14B-Instruct"
)
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

log "=== D2 容量曲线启动: 等 GPU 空闲 ==="
while [ "$(gpu_busy)" = "busy" ]; do log "GPU忙,等待前序任务..."; sleep 180; done

DATA15="$PWD/15_fulldata_resplit/data"
STAGE1="$PWD/shared/train_choice_head_distill.py"

for size in 0.5B 1.5B 3B 7B 14B; do
  marker="SIZE_${size}_DONE"
  if grep -q "$marker" "$LOG" 2>/dev/null; then log "$size 已完成,跳过"; continue; fi

  repo="${REPO[$size]}"
  # 本地路径 or 下载
  if [[ "$repo" == /* ]]; then
    model_path="$repo"
  else
    model_path="$PWD/models/Qwen2.5-${size}-Instruct"
    if [ ! -d "$model_path" ]; then
      log "下载 $size ($repo)..."
      "$EASYEDIT_PY" -c "from huggingface_hub import snapshot_download; snapshot_download('$repo', local_dir='$model_path')" \
        >> "$LOG" 2>&1 || { log "WARN 下载 $size 失败,跳过"; continue; }
    fi
  fi

  while [ "$(gpu_busy)" = "busy" ]; do sleep 120; done
  out_dir="$DIST/runs/d2_capacity/qwen${size}_seed42/stage1_head"
  mkdir -p "$out_dir"
  log "--- 蒸馏学生 $size (教师 Qwen32B, baseline_all 全量) ---"
  "$EASYEDIT_PY" "$STAGE1" \
    --model_name "$model_path" \
    --data_path "$DS_DIR/train_baseline_all.jsonl" \
    --val_path "$DATA15/val.jsonl" \
    --test_path "$DATA15/test.jsonl" \
    --output_dir "$out_dir" \
    --num_epochs 1 --batch_size 2 --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 --rank 16 --lora_alpha 32 --alpha 0.35 \
    --default_distill_mask 0 --seed 42 --deterministic \
    >> "$DIST/d2_size_${size}.log" 2>&1 && log "$marker" || log "WARN $size 蒸馏失败"
done

# 拟合曲线 + 反查容量下限
log "=== 建容量-正确率曲线 ==="
"$EASYEDIT_PY" "$DIST/scripts/d2_fit_curve.py" > "$DIST/outputs/d2_curve.log" 2>&1 || log "WARN 拟合失败"
log "=== D2 完成 ==="
echo "D2_DONE" >> "$LOG"
