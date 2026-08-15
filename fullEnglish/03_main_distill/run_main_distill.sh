#!/usr/bin/env bash
# fullEnglish — 主蒸馏实验 (学生 = Qwen2.5-32B, 主教师 = DeepSeek/Gemini/Llama).
# 头条臂: α=0 (纯 GT SFT / 决策空间监督, 中文+英文牙科已证最优) × 3 seed.
# 可选 α 扫描 (RUN_ALPHA_SWEEP=1): α∈{0.35, 1.0} × 1 seed, 复现「KL 越多越差」跨语言结论.
# 选点=val(MedQA dev), 报告=test_medqa / test_medmcqa / test_mmlu / test_pubmedqa(泛化).
# 全程单 H100 顺序跑. 目标: 判定全医科英文数据下学生能否超越教师.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # -> mentalDistill/
source setup.env 2>/dev/null || true
PY="${EASYEDIT_PY:-$HOME/anaconda3/bin/python3}"
FE="fullEnglish/03_main_distill"
DATA="fullEnglish/00_data/out"
RUN="$FE/runs"
STUDENT="${STUDENT_MODEL:-$BASE_MODEL_32B}"
STUDENT="${STUDENT:-models/Qwen2.5-32B-Instruct}"
SEEDS=(11 42 8)
export DISTILL_PROMPT_LANG=en   # 训练 prompt 用英文

wait_gpu_idle () {
  while :; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    if [[ -n "${used:-}" && "$used" -lt 20000 ]]; then return 0; fi
    echo "[$(date +%H:%M:%S)] GPU 忙 (${used:-?} MiB), 等待其他训练释放..."
    sleep 120
  done
}

mkdir -p "$RUN" "$FE/data"
if [[ ! -d "$STUDENT" ]]; then echo "[FATAL] 学生模型缺失: $STUDENT"; exit 1; fi

echo "==================================================================="
echo "fullEnglish 主蒸馏 — 学生=$STUDENT"
echo "train=$(wc -l < "$DATA/train.jsonl")  val=$(wc -l < "$DATA/val.jsonl")  (MedQA dev)"
echo "==================================================================="

########## STEP 0: 主教师标签 (训练集软标签 + 测试集同集准确率) ##########
bash "$FE/generate_teacher_labels.sh"

########## STEP 1: 构造 Choice-Head 训练文件 ##########
"$PY" "$FE/build_train_head.py" \
    --train "$DATA/train.jsonl" \
    --teacher "$FE/labels/teacher_train.jsonl" \
    --output "$FE/data/train_head_distill.jsonl" \
    2>&1 | tee "$RUN/build_data.log"

########## STEP 2: 学生零样本地板 (无训练) ##########
echo "########## 学生零样本地板 (32B, 无 adapter) ##########"
wait_gpu_idle
for t in test_medqa test_medmcqa test_mmlu test_pubmedqa; do
  "$PY" fullEnglish/04_eval/eval_mcq.py --base_model "$STUDENT" \
      --test_data "$DATA/$t.jsonl" --label "zeroshot_$t" \
      2>&1 | tee "$RUN/zeroshot_$t.log"
done

########## STEP 3: 训练 (头条 α=0 × 3 seed) ##########
run_arm () {
  local alpha=$1 tag=$2 seeds=("${@:3}")
  for seed in "${seeds[@]}"; do
    local name="32B_${tag}_s${seed}"
    local out="$RUN/$name"
    if [[ -f "$out/DONE" ]]; then echo "[SKIP] $name"; continue; fi
    mkdir -p "$out"
    wait_gpu_idle
    echo "-------------------------------------------------------------------"
    echo "[$(date +%H:%M:%S)] TRAIN $name  (alpha=$alpha seed=$seed)"
    "$PY" shared/train_choice_head_distill.py \
      --model_name "$STUDENT" \
      --data_path "$FE/data/train_head_distill.jsonl" \
      --val_path "$DATA/val.jsonl" \
      --output_dir "$out" --num_epochs 1 --batch_size 1 --gradient_accumulation_steps 8 \
      --learning_rate 1e-4 --rank 16 --lora_alpha 32 --alpha "$alpha" \
      --default_distill_mask 1 --seed "$seed" --deterministic \
      > "$out/train.log" 2>&1
    touch "$out/DONE"
    echo "[$(date +%H:%M:%S)] done $name"
  done
}

echo "########## STEP 3a: α=0 (HEADLINE: 纯 CE / 决策空间监督) ##########"
run_arm 0.0 a00 "${SEEDS[@]}"

if [[ "${RUN_ALPHA_SWEEP:-0}" == "1" ]]; then
  echo "########## STEP 3b: α=0.35 (KL 蒸馏对照) ##########"
  run_arm 0.35 a35 11
  echo "########## STEP 3c: α=1.0 (纯 KL, 最坏锚) ##########"
  run_arm 1.0 a10 11
fi

########## STEP 4: 评估所有 adapter + 聚合 ##########
echo "########## STEP 4: 评估 + 聚合 ##########"
wait_gpu_idle
"$PY" "$FE/evaluate_all.py" 2>&1 | tee "$RUN/RESULTS.log"
"$PY" "$FE/aggregate_results.py" 2>&1 | tee -a "$RUN/RESULTS.log"

echo "[$(date +%H:%M:%S)] fullEnglish 主蒸馏完成"
