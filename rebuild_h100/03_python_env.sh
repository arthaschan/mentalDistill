#!/usr/bin/env bash
# 03 Python 环境：base 装训练依赖 + 独立 vllm conda 环境。
set -euo pipefail
echo "======== 03 Python 环境 ========"
CONDA="$HOME/anaconda3/bin/conda"
PY="$HOME/anaconda3/bin/python3"
REPO="$HOME/arthas/mentalDistill"

# --- base 环境：训练依赖 ---
echo "→ [base] 安装训练依赖（torch cu128 + transformers 等）"
"$PY" -m pip install --upgrade pip
# torch 2.9.x + CUDA 12.8（匹配 H100 驱动 575）
"$PY" -m pip install "torch==2.9.1" --index-url https://download.pytorch.org/whl/cu128 || \
  "$PY" -m pip install torch --index-url https://download.pytorch.org/whl/cu128
# 项目 requirements + 关键固定版本
if [ -f "$REPO/requirements.txt" ]; then "$PY" -m pip install -r "$REPO/requirements.txt"; fi
"$PY" -m pip install \
  "transformers==4.57.6" "peft==0.7.1" "datasets==3.6.0" \
  accelerate sentence-transformers huggingface_hub tqdm requests numpy

# --- 独立 vllm 环境（隔离 autoawq/transformers 冲突）---
echo "→ [vllm] 建独立 conda 环境（供 Llama-70B-AWQ 等大教师）"
if ! "$CONDA" env list | grep -q "^vllm "; then
  "$CONDA" create -y -n vllm python=3.11
fi
"$CONDA" run -n vllm pip install --upgrade pip
"$CONDA" run -n vllm pip install "vllm==0.16.0" || "$CONDA" run -n vllm pip install vllm
echo "======== 03 完成 ========"
echo "验证: $PY -c 'import torch;print(torch.__version__, torch.cuda.is_available())'"
