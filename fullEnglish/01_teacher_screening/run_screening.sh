#!/usr/bin/env bash
# fullEnglish — 教师预评估 (Teacher Screening, 零训练成本).
# 在 screen_input.jsonl (600 题, MedQA/MedMCQA/MMLU 各 200) 上跑 zero-shot,
# 产出「教师能力先验表」+ 学生零样本地板 → 判断 headroom (能否超越).
# API 教师 (DeepSeek/Gemini/Llama) 走 generate_teacher_labels_api.py;
# 本地教师 (Qwen32B=学生base / Qwen14B / Gemma27B / Phi4 / Yi34B) 走 logprobs;
# Llama-70B-AWQ 本地 vLLM 可选 (无需 API key).
# 顺序执行, 单 H100. 缺 key / 缺模型自动跳过.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # -> mentalDistill/
source setup.env 2>/dev/null || true
PY="${EASYEDIT_PY:-$HOME/anaconda3/bin/python3}"
VLLM_PY="$HOME/anaconda3/envs/vllm/bin/python"
FE="fullEnglish/01_teacher_screening"
DS="fullEnglish/00_data/out/screen_input.jsonl"
SP="$FE/system_prompt_en.txt"
TRAIL="$FE/trailing_instruction_en.txt"
mkdir -p "$FE/labels" "$FE/logprobs" "$FE/reports"

echo "==================================================================="
echo "fullEnglish 教师预评估 — 屏幕输入: $DS ($(wc -l < "$DS") 题)"
echo "==================================================================="

########################################
# 1. API 教师 (DeepSeek / Gemini / Llama)
########################################
api_label () {
  local name="$1" cand="$2" interval="${3:-0.5}"
  local key_env
  key_env=$(python3 -c "import json;print(json.load(open('$cand')).get('api_key_env',''))")
  local key_val="${!key_env:-}"
  if [[ -z "$key_val" ]]; then
    echo "[SKIP] $name: 环境变量 $key_env 未设置 (缺 API key)"
    return
  fi
  echo "-------------------------------------------------------------------"
  echo "[$(date +%H:%M:%S)] API teacher=$name  candidate=$cand  interval=${interval}s"
  "$PY" shared/generate_teacher_labels_api.py \
      --candidate "$cand" --system_prompt "$SP" --trailing_instruction_file "$TRAIL" \
      --dataset "$DS" --output "$FE/labels/${name}.jsonl" --resume \
      --request_interval_sec "$interval" --max_retries 5 --rate_limit_cooldown_sec 60 \
      2>&1 | tail -8
  echo "[$(date +%H:%M:%S)] done $name -> labels/${name}.jsonl"
}

api_label DeepSeekV3 "$FE/candidates/deepseek.json" 0.5
# Gemini (用户要最好的=pro): 先探测可达性. 免费额度超限(429)/地区受限(400)则快速跳过,
# 避免进入"每题重试5次"浪费数小时.
if [[ -n "${GEMINI_API_KEY:-}" && "${GEMINI_API_KEY}" != "填这里" ]]; then
  probe=$(curl -s -m 15 "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions" \
    -H "Authorization: Bearer $GEMINI_API_KEY" -H "Content-Type: application/json" \
    -d '{"model":"gemini-pro-latest","messages":[{"role":"user","content":"hi"}],"max_tokens":2}' 2>/dev/null)
  if echo "$probe" | grep -q '"error"'; then
    echo "[SKIP] Gemini 探测失败 (免费额度超限/地区受限): $(echo "$probe" | head -c 160)"
  else
    api_label GeminiPro "$FE/candidates/gemini.json" 12
  fi
else
  echo "[SKIP] Gemini: 无 key"
fi
# Llama 云端: 优先硅基流动(国内免代理), 其次 OpenRouter / Groq (需代理).
if [[ -n "${SILICONFLOW_API_KEY:-}" && "${SILICONFLOW_API_KEY}" != "填这里" ]]; then
  api_label Llama70B "$FE/candidates/llama_siliconflow.json" 1
elif [[ -n "${OPENROUTER_API_KEY:-}" && "${OPENROUTER_API_KEY}" != "填这里" ]]; then
  api_label Llama70B "$FE/candidates/llama_openrouter.json" 1
elif [[ -n "${GROQ_API_KEY:-}" && "${GROQ_API_KEY}" != "填这里" ]]; then
  api_label Llama70B "$FE/candidates/llama_groq.json" 1
else
  echo "[SKIP] Llama70B(云端 API): 无可用 key. 本地 vLLM 分支会在下面跑 (无需 API)."
fi

########################################
# 2. 本地教师 (真实 logprobs, 无需 key)
########################################
# Qwen32B 同时是「学生 base」→ 它的 zero-shot 就是学生零样本地板.
declare -A TEACHERS=(
  [Qwen32B]="models/Qwen2.5-32B-Instruct"
  [Qwen14B]="models/Qwen2.5-14B-Instruct"
  [Gemma27B]="models/gemma-2-27b-it"
  [Phi4]="models/phi-4"
  [Yi34B]="models/Yi-1.5-34B-Chat"
)
ORDER=(Phi4 Gemma27B Yi34B Qwen14B Qwen32B)
for name in "${ORDER[@]}"; do
  mp="${TEACHERS[$name]}"
  if [[ ! -d "$mp" ]]; then echo "[SKIP] $name: 模型目录缺失 $mp"; continue; fi
  echo "-------------------------------------------------------------------"
  echo "[$(date +%H:%M:%S)] LOCAL teacher=$name  model=$mp"
  "$PY" shared/generate_teacher_labels_local_logprobs.py \
      --model_path "$mp" --dataset "$DS" \
      --output "$FE/logprobs/${name}_logprobs.jsonl" \
      --system_prompt "$SP" --trailing_instruction_file "$TRAIL" \
      --gt_field Answer --resume 2>&1 | tail -6
  echo "[$(date +%H:%M:%S)] done $name -> logprobs/${name}_logprobs.jsonl"
done

########################################
# 3. (可选) Llama-70B-AWQ 本地 vLLM — 无需 API key
########################################
if [[ -x "$VLLM_PY" && -d "models/Llama-3.3-70B-Instruct-AWQ" ]]; then
  echo "-------------------------------------------------------------------"
  echo "[$(date +%H:%M:%S)] LOCAL teacher=Llama70B-AWQ  (vLLM)"
  "$VLLM_PY" shared/generate_teacher_labels_vllm.py \
      --model_path "models/Llama-3.3-70B-Instruct-AWQ" --dataset "$DS" \
      --output "$FE/logprobs/Llama70B-AWQ_logprobs.jsonl" \
      --system_prompt "$SP" --trailing_instruction_file "$TRAIL" \
      --gt_field Answer --resume 2>&1 | tail -6
else
  echo "[SKIP] Llama70B-AWQ(vLLM): 无 vllm 环境或无模型"
fi

########################################
# 4. 聚合 -> 教师先验表 + headroom
########################################
echo "==================================================================="
echo "[$(date +%H:%M:%S)] 聚合教师先验..."
"$PY" "$FE/aggregate_screening.py"
echo "[$(date +%H:%M:%S)] 教师预评估完成"
