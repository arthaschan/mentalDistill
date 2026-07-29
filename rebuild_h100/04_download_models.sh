#!/usr/bin/env bash
# 04 从 HuggingFace 下载 12 个模型到 models/（共约 382GB，最耗时）。
# 支持断点续传。已有的可注释掉。需要时先 huggingface-cli login。
set -uo pipefail
echo "======== 04 下载模型（382GB，耐心等）========"
PY="$HOME/anaconda3/bin/python3"
REPO="$HOME/arthas/mentalDistill"
MODELS="$REPO/models"
mkdir -p "$MODELS"
DL="$PY -m huggingface_hub.commands.huggingface_cli download"
command -v huggingface-cli >/dev/null 2>&1 && DL="huggingface-cli download"

# 若部分模型需登录: huggingface-cli login  (gemma/llama 等 gated 需先申请权限)
# 格式: <repo_id> -> models/<本地目录名>
declare -A M=(
  ["Qwen/Qwen2.5-0.5B-Instruct"]="Qwen2.5-0.5B-Instruct"
  ["Qwen/Qwen2.5-1.5B-Instruct"]="Qwen2.5-1.5B-Instruct"
  ["Qwen/Qwen2.5-3B-Instruct"]="Qwen2.5-3B-Instruct"
  ["Qwen/Qwen2.5-7B-Instruct"]="Qwen2.5-7B-Instruct"
  ["Qwen/Qwen2.5-14B-Instruct"]="Qwen2.5-14B-Instruct"
  ["Qwen/Qwen2.5-32B-Instruct"]="Qwen2.5-32B-Instruct"
  ["Qwen/Qwen3-14B"]="Qwen3-14B"
  ["THUDM/GLM-4-32B-0414"]="GLM-4-32B-0414"
  ["google/gemma-2-27b-it"]="gemma-2-27b-it"
  ["microsoft/phi-4"]="phi-4"
  ["01-ai/Yi-1.5-34B-Chat"]="Yi-1.5-34B-Chat"
  ["casperhansen/llama-3.3-70b-instruct-awq"]="Llama-3.3-70B-Instruct-AWQ"
)
for repo in "${!M[@]}"; do
  dst="$MODELS/${M[$repo]}"
  if [ -d "$dst" ] && [ -n "$(ls -A "$dst" 2>/dev/null)" ]; then
    echo "✓ 跳过(已存在): ${M[$repo]}"; continue
  fi
  echo "↓ 下载 $repo -> $dst"
  $DL "$repo" --local-dir "$dst" --local-dir-use-symlinks False || \
    echo "✗ $repo 下载失败（可能需 huggingface-cli login 或申请 gated 权限；稍后重跑续传）"
done
echo "======== 04 完成 ========"
du -sh "$MODELS" 2>/dev/null
