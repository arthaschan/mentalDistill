#!/usr/bin/env bash
# 00 系统检查（只读，不改动）。确认 H100 机器就绪。
set -uo pipefail
echo "======== H100 重建 · 系统检查 ========"
echo "--- 主机/内核 ---"; uname -a
echo "--- GPU ---"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  NGPU=$(nvidia-smi -L | wc -l)
  echo "物理 GPU 数量: $NGPU  (预期 1 张 H100 NVL 95GB)"
else
  echo "✗ 未找到 nvidia-smi —— 驱动未装或未加载"
fi
echo "--- 磁盘剩余（需 >=600GB 给模型+产物）---"
df -h "$HOME" | awk 'NR==1||NR==2'
echo "--- Python / conda ---"
command -v conda >/dev/null 2>&1 && conda --version || echo "conda: 未装（01 脚本会装）"
[ -x "$HOME/anaconda3/bin/python3" ] && "$HOME/anaconda3/bin/python3" --version || echo "anaconda python: 未装"
echo "--- 网络连通性 ---"
for h in github.com huggingface.co astral.sh pypi.org; do
  timeout 5 bash -c "</dev/tcp/$h/443" 2>/dev/null && echo "✓ $h 可达" || echo "✗ $h 不可达"
done
echo "======== 检查完成 ========"
