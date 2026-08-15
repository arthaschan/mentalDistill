#!/usr/bin/env bash
# fullEnglish — 生成主教师标签 (训练集 + 三个测试集).
# TEACHER_MODE=api  -> generate_teacher_labels_api.py (硬标签, 需 API key)
# TEACHER_MODE=local-> generate_teacher_labels_local_logprobs.py (真实 logprobs)
# 输出到 03_main_distill/labels/teacher_{train,test_medqa,test_medmcqa,test_mmlu}.jsonl
# 测试集标签同时用于「教师同集准确率」(与学生对标).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # -> mentalDistill/
source setup.env 2>/dev/null || true
PY="${EASYEDIT_PY:-$HOME/anaconda3/bin/python3}"
FE="fullEnglish/03_main_distill"
DATA="fullEnglish/00_data/out"
SP="fullEnglish/01_teacher_screening/system_prompt_en.txt"
TRAIL="fullEnglish/01_teacher_screening/trailing_instruction_en.txt"
MODE="${TEACHER_MODE:-api}"
mkdir -p "$FE/labels"

echo "=== 主教师标签生成 (mode=$MODE) ==="

if [[ "$MODE" == "api" ]]; then
  CAND="${TEACHER_CANDIDATE:-$FE/teacher_candidate.json}"
  KEY_ENV=$(python3 -c "import json;print(json.load(open('$CAND')).get('api_key_env',''))")
  KEY_VAL="${!KEY_ENV:-}"
  if [[ -z "$KEY_VAL" ]]; then
    echo "[FATAL] API 教师缺 key: 环境变量 $KEY_ENV 未设置. 设 TEACHER_MODE=local 或填 setup.env"
    exit 1
  fi
  LABEL_CMD=("$PY" shared/generate_teacher_labels_api.py
             --candidate "$CAND" --system_prompt "$SP" --trailing_instruction_file "$TRAIL"
             --request_interval_sec 0.2 --max_retries 6)
  TAG=$(python3 -c "import json;print(json.load(open('$CAND')).get('name','teacher'))")
else
  TM="${TEACHER_MODEL:-models/Qwen2.5-32B-Instruct}"
  if [[ ! -d "$TM" ]]; then echo "[FATAL] 本地教师模型缺失: $TM"; exit 1; fi
  LABEL_CMD=("$PY" shared/generate_teacher_labels_local_logprobs.py
             --model_path "$TM" --system_prompt "$SP" --trailing_instruction_file "$TRAIL" --gt_field Answer)
  TAG=$(basename "$TM")
fi

for spec in "train:$DATA/train.jsonl" "test_medqa:$DATA/test_medqa.jsonl" \
            "test_medmcqa:$DATA/test_medmcqa.jsonl" "test_mmlu:$DATA/test_mmlu.jsonl"; do
  name="${spec%%:*}"; ds="${spec#*:}"
  echo "--- [$TAG] 标注 $name: $ds ($(wc -l < "$ds") 题)"
  "${LABEL_CMD[@]}" --dataset "$ds" --output "$FE/labels/teacher_${name}.jsonl" --resume 2>&1 | tail -4
done

# 教师同集准确率 (每个测试集)
python3 - "$FE/labels" <<'PYEOF'
import json, sys, glob, os
d = sys.argv[1]
print("\n=== 教师同集准确率 (学生同题对标锚) ===")
for f in ["teacher_test_medqa.jsonl","teacher_test_medmcqa.jsonl","teacher_test_mmlu.jsonl"]:
    p = os.path.join(d, f)
    if not os.path.exists(p):
        print(f"  {f}: (缺失)"); continue
    n=c=0
    for line in open(p):
        line=line.strip()
        if not line: continue
        r=json.loads(line)
        gt=str(r.get("OriginalAnswer") or r.get("Answer","")).strip().upper()
        ta=str(r.get("TeacherAnswer") or r.get("Answer","")).strip().upper()
        if gt in "ABCDE" and ta in "ABCDE":
            n+=1; c+= int(ta==gt)
    print(f"  {f:28s} {100*c/n:.2f}% ({c}/{n})" if n else f"  {f}: 0")
PYEOF

echo "=== 教师标签生成完成: $FE/labels/"
