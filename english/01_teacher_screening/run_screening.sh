#!/usr/bin/env bash
# Teacher screening on English dental single-best pool (636 items).
# Sequential on ONE H100. Generates ABCDE logprobs per teacher, then aggregates.
set -uo pipefail
cd /home/student/arthas/mentalDistill
source setup.env 2>/dev/null || true
PY="${EASYEDIT_PY:-/home/student/anaconda3/bin/python3}"

DS="english/01_teacher_screening/screen_input.jsonl"
PROMPT="english/01_teacher_screening/system_prompt_en.txt"
LPDIR="english/01_teacher_screening/logprobs"
REPDIR="english/01_teacher_screening/reports"
mkdir -p "$LPDIR" "$REPDIR"

# teacher pool (label : model_dir). Strong/mid/weak cross-family, mirrors prior 6-teacher config.
declare -A TEACHERS=(
  [Qwen32B]="models/Qwen2.5-32B-Instruct"
  [Qwen14B]="models/Qwen2.5-14B-Instruct"
  [GLM32B]="models/GLM-4-32B-0414"
  [Yi34B]="models/Yi-1.5-34B-Chat"
  [Gemma27B]="models/gemma-2-27b-it"
  [Phi4]="models/phi-4"
  [Llama70B]="models/Llama-3.3-70B-Instruct-AWQ"
  [Qwen7B]="models/Qwen2.5-7B-Instruct"
)
# screening order: small->large so early feedback comes fast
ORDER=(Phi4 Qwen7B Gemma27B Yi34B Qwen14B GLM32B Qwen32B Llama70B)

for name in "${ORDER[@]}"; do
  mp="${TEACHERS[$name]}"
  lp="$LPDIR/${name}_logprobs.jsonl"
  echo "==================================================================="
  echo "[$(date +%H:%M:%S)] TEACHER=$name  model=$mp"
  if [[ ! -d "$mp" ]]; then echo "  [SKIP] model dir missing: $mp"; continue; fi
  "$PY" shared/generate_teacher_labels_local_logprobs.py \
      --model_path "$mp" --dataset "$DS" --output "$lp" \
      --system_prompt "$PROMPT" --gt_field Answer --resume \
      2>&1 | tail -12
  echo "[$(date +%H:%M:%S)] done $name -> $lp"
done

echo "==================================================================="
echo "[$(date +%H:%M:%S)] all teachers done. aggregating..."
"$PY" english/01_teacher_screening/aggregate_screening.py
echo "[$(date +%H:%M:%S)] SCREENING COMPLETE"
