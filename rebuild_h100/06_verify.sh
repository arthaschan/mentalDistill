#!/usr/bin/env bash
# 06 验证重建成功。逐项应为 ✓。
set -uo pipefail
echo "======== 06 验证重建 ========"
PY="$HOME/anaconda3/bin/python3"
REPO="$HOME/arthas/mentalDistill"
ok(){ echo "✓ $1"; }; bad(){ echo "✗ $1"; }

# GPU
nvidia-smi >/dev/null 2>&1 && ok "GPU 可见 ($(nvidia-smi --query-gpu=name --format=csv,noheader|head -1))" || bad "GPU 不可见"
# torch CUDA
"$PY" -c 'import torch;assert torch.cuda.is_available()' 2>/dev/null \
  && ok "torch CUDA 可用 ($($PY -c "import torch;print(torch.__version__)"))" || bad "torch CUDA 不可用"
# 关键包
"$PY" -c 'import transformers,peft,datasets,sentence_transformers' 2>/dev/null && ok "训练依赖齐全" || bad "训练依赖缺失"
# vllm 环境
"$HOME/anaconda3/bin/conda" run -n vllm python -c 'import vllm' 2>/dev/null && ok "vllm 环境可用" || bad "vllm 环境缺失"
# 模型
if [ -d "$REPO/models" ]; then
  N=$(ls "$REPO/models" 2>/dev/null | wc -l); echo "→ models/ 下 $N 个模型 (预期 12)"
  for m in Qwen2.5-7B-Instruct Qwen2.5-14B-Instruct; do
    [ -d "$REPO/models/$m" ] && ok "模型 $m" || bad "模型 $m 缺失"
  done
else bad "models/ 不存在（跑 04）"; fi
# setup.env
[ -f "$REPO/setup.env" ] && ok "setup.env 存在（记得填 API keys）" || echo "→ setup.env 未建，从 rebuild_h100/setup.env.template 复制"
# hermes
if command -v hermes >/dev/null 2>&1; then hermes --version 2>/dev/null && ok "hermes 可启动" || bad "hermes 启动异常"
else bad "hermes 命令找不到（检查 PATH 含 ~/.local/bin）"; fi
echo "======== 验证完成 ========"
