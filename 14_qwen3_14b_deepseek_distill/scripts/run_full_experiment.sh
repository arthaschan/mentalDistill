#!/usr/bin/env bash
# =============================================================================
# Qwen3-14B 完整实验管线 (后台运行)
# Phase 0: 零样本评估
# Phase 1 (C1): 真实 multivote 软标签 + 过滤 → 单阶段蒸馏
# Phase 2 (C3): CoT 推理链蒸馏
# =============================================================================
set -euo pipefail

export ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PROJECT_ROOT="$(cd "$ROOT_DIR/.." && pwd)"
SHARED="$PROJECT_ROOT/shared"
LOGDIR="$ROOT_DIR/logs"
mkdir -p "$LOGDIR"

source "$PROJECT_ROOT/setup.env" 2>/dev/null || true
export MODEL="${BASE_MODEL_QWEN3_14B:-/home/student/arthas/EasyEdit3/Qwen3-14B}"
PY="${EASYEDIT_PY:-python3}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="$LOGDIR/full_experiment_${TIMESTAMP}.log"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$MASTER_LOG"; }

log "=== Qwen3-14B 完整实验管线启动 ==="
log "模型: $MODEL"
log "日志: $MASTER_LOG"

# ============= Phase 0: 零样本评估 =============
log ""
log "===== Phase 0: 零样本基线评估 ====="

$PY -u << 'PYEOF' 2>&1 | tee -a "$MASTER_LOG"
import json, torch, os

OPTION_LETTERS = ["A", "B", "C", "D", "E"]
def extract_answer_char(text):
    for ch in text.strip().upper():
        if ch in OPTION_LETTERS: return ch
    return ""

model_path = os.environ["MODEL"]
test_data = os.path.join(os.environ["ROOT_DIR"], "data/test.jsonl")

from transformers import AutoModelForCausalLM, AutoTokenizer

print(f"加载模型: {model_path}", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model.eval()

samples = []
with open(test_data) as f:
    for line in f:
        if line.strip(): samples.append(json.loads(line))

# Non-thinking mode
correct = 0; total = 0
for item in samples:
    gt = item.get("Answer","").strip().upper()
    if not gt: continue
    total += 1
    q, opts = item.get("Question",""), item.get("Options","")
    prompt = (f"<|im_start|>system\n你是一位专业的牙科医生。请根据你的专业知识回答以下选择题，只输出一个大写字母（A/B/C/D/E）。<|im_end|>\n"
              f"<|im_start|>user\n{q}\n{opts}\n请只输出一个大写字母作为答案。<|im_end|>\n"
              f"<|im_start|>assistant\n<think>\n\n</think>\n\n")
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=8, do_sample=False,
                           pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    pred = extract_answer_char(resp)
    if pred == gt[0].upper(): correct += 1

acc = correct / total * 100
print(f"\n[PHASE0-RESULT] Qwen3-14B 零样本 (non-thinking): {acc:.2f}% ({correct}/{total})", flush=True)

del model; torch.cuda.empty_cache()
print("[PHASE0] GPU 已释放", flush=True)
PYEOF

log "Phase 0 完成"

# ============= Phase 1 (C1): 真实 multivote 软标签单阶段蒸馏 =============
log ""
log "===== Phase 1 (C1): 真实 multivote 标签 + 过滤有害样本 → 单阶段蒸馏 ====="

# Step 1.1: 构建清洁训练数据
log "Step 1.1: 构建清洁 head distill 数据集"
$PY -u << 'PYEOF' 2>&1 | tee -a "$MASTER_LOG"
import json, os

ROOT = os.environ["ROOT_DIR"]
PROJ = os.environ["PROJECT_ROOT"]

# 加载 GT 训练数据
gt_data = {}
with open(os.path.join(ROOT, "data/train.jsonl")) as f:
    for line in f:
        if not line.strip(): continue
        item = json.loads(line)
        key = item["Question"].strip()[:80]
        gt_data[key] = item

# 加载真实 multivote 标签
mv_path = os.path.join(PROJ, "02_deepseek_v3_choice_head/data/teacher_train_multivote.jsonl")
mv_data = {}
with open(mv_path) as f:
    for line in f:
        if not line.strip(): continue
        item = json.loads(line)
        key = item["Question"].strip()[:80]
        mv_data[key] = item

# 合并: 只保留教师多数投票 == GT 的样本用蒸馏，其余用纯 CE
output = []
stats = {"total": 0, "matched": 0, "clean_teacher": 0, "teacher_wrong": 0, "gt_only": 0}

for key, gt_item in gt_data.items():
    stats["total"] += 1
    out_item = dict(gt_item)

    if key in mv_data:
        stats["matched"] += 1
        mv_item = mv_data[key]
        dist = mv_item.get("TeacherDist", {})
        gt_ans = gt_item["Answer"].strip().upper()

        if dist:
            teacher_ans = max(dist, key=dist.get)
            if teacher_ans == gt_ans:
                out_item["TeacherDist"] = dist
                out_item["SelectiveSource"] = "clean_teacher"
                stats["clean_teacher"] += 1
            else:
                out_item["TeacherDist"] = {}
                out_item["SelectiveSource"] = "gt_fallback"
                stats["teacher_wrong"] += 1
        else:
            out_item["TeacherDist"] = {}
            out_item["SelectiveSource"] = "gt_fallback"
            stats["gt_only"] += 1
    else:
        out_item["TeacherDist"] = {}
        out_item["SelectiveSource"] = "gt_fallback"
        stats["gt_only"] += 1

    output.append(out_item)

out_path = os.path.join(ROOT, "data/train_head_distill_clean.jsonl")
with open(out_path, "w", encoding="utf-8") as f:
    for item in output:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"[C1-DATA] 构建完成: {out_path}")
print(f"  总计: {stats['total']}")
print(f"  匹配 multivote: {stats['matched']}")
print(f"  clean_teacher (教师正确): {stats['clean_teacher']}")
print(f"  teacher_wrong (已过滤): {stats['teacher_wrong']}")
print(f"  gt_only (无教师标签): {stats['gt_only']}")
PYEOF

# Step 1.2: 单阶段 choice-head 蒸馏 (3 seeds, 1 epoch only)
log "Step 1.2: 单阶段 choice-head 蒸馏 (3 seeds)"

C1_RUN="$ROOT_DIR/runs/${TIMESTAMP}_c1_clean_labels"
mkdir -p "$C1_RUN/logs" "$C1_RUN/outputs"

for SEED in 42 11 8; do
  TAG="c1_s${SEED}"
  OUT="$C1_RUN/outputs/$TAG/stage1_head"
  mkdir -p "$OUT"

  log "  训练 seed=$SEED → $OUT"
  $PY "$SHARED/train_choice_head_distill.py" \
    --model_name "$MODEL" \
    --data_path "$ROOT_DIR/data/train_head_distill_clean.jsonl" \
    --val_path "$ROOT_DIR/data/val.jsonl" \
    --test_path "$ROOT_DIR/data/test.jsonl" \
    --output_dir "$OUT" \
    --num_epochs 1 \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --rank 16 \
    --lora_alpha 32 \
    --alpha 0.35 \
    --default_distill_mask 0 \
    --seed $SEED \
    --deterministic \
    2>&1 | tee "$C1_RUN/logs/stage1_${TAG}.log"

  # 提取结果
  RESULT=$(grep -oP "测试集准确率.*" "$C1_RUN/logs/stage1_${TAG}.log" | tail -1 || echo "未找到结果")
  log "  [C1-RESULT] seed=$SEED: $RESULT"
done

log "Phase 1 (C1) 完成"

# ============= Phase 2 (C3): CoT 推理链蒸馏 =============
log ""
log "===== Phase 2 (C3): CoT 推理链蒸馏 ====="

COT_SRC="$PROJECT_ROOT/12_rationale_distill/data/train_cot.jsonl"
if [[ ! -f "$COT_SRC" ]]; then
  log "[C3] 错误: 找不到 CoT 数据 $COT_SRC"
  log "跳过 Phase 2"
else
  C3_RUN="$ROOT_DIR/runs/${TIMESTAMP}_c3_cot_distill"
  mkdir -p "$C3_RUN/logs" "$C3_RUN/outputs"

  # 构建 CoT SFT 训练数据 (只保留教师答对的样本)
  log "Step 2.1: 构建 CoT SFT 训练数据"
  $PY -u << 'PYEOF' 2>&1 | tee -a "$MASTER_LOG"
import json, os

ROOT = os.environ["ROOT_DIR"]
cot_src = os.environ["PROJECT_ROOT"] + "/12_rationale_distill/data/train_cot.jsonl"

items = []
with open(cot_src) as f:
    for line in f:
        if not line.strip(): continue
        item = json.loads(line)
        if item.get("GTMatch"):
            items.append(item)

print(f"CoT 样本数 (GT match): {len(items)}")

out_path = os.path.join(ROOT, "data/train_cot_sft.jsonl")
with open(out_path, "w", encoding="utf-8") as f:
    for item in items:
        entry = {
            "Question": item["Question"],
            "Options": item["Options"],
            "Answer": item["Answer"],
        }
        if "Rationale" in item:
            entry["Rationale"] = item["Rationale"]
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"[C3-DATA] 写入 {out_path}, 共 {len(items)} 条")
PYEOF

  for SEED in 42 11; do
    TAG="c3_cot_s${SEED}"
    OUT="$C3_RUN/outputs/$TAG"
    mkdir -p "$OUT"

    log "  CoT SFT 训练 seed=$SEED"
    $PY "$SHARED/train_gt_sft.py" \
      --model_name "$MODEL" \
      --data_path "$ROOT_DIR/data/train_cot_sft.jsonl" \
      --val_path "$ROOT_DIR/data/val.jsonl" \
      --test_path "$ROOT_DIR/data/test.jsonl" \
      --output_dir "$OUT" \
      --num_epochs 3 \
      --batch_size 2 \
      --gradient_accumulation_steps 4 \
      --learning_rate 1e-4 \
      --rank 16 \
      --lora_alpha 32 \
      --alpha 0.0 \
      --default_distill_mask 0 \
      --seed $SEED \
      --deterministic \
      2>&1 | tee "$C3_RUN/logs/${TAG}.log"

    RESULT=$(grep -oP "测试集准确率.*" "$C3_RUN/logs/${TAG}.log" | tail -1 || echo "未找到结果")
    log "  [C3-RESULT] seed=$SEED: $RESULT"
  done

  log "Phase 2 (C3) 完成"
fi

# ============= 汇总 =============
log ""
log "========================================="
log "          实验全部完成"
log "========================================="
log ""
log "=== Phase 0: 零样本 ==="
grep "PHASE0-RESULT" "$MASTER_LOG" || true
log ""
log "=== Phase 1 (C1): 清洁标签单阶段蒸馏 ==="
grep "C1-RESULT" "$MASTER_LOG" || true
log ""
log "=== Phase 2 (C3): CoT SFT ==="
grep "C3-RESULT" "$MASTER_LOG" || true
log ""
log "详细日志: $MASTER_LOG"
log "========================================="
