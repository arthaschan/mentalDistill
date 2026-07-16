#!/usr/bin/env bash
# 01 装 Anaconda(若无) + 系统工具 + git-lfs。幂等。
set -euo pipefail
echo "======== 01 基础环境 ========"

# 系统工具（需要 sudo；无 sudo 则跳过并提示）
if command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
  sudo apt-get update -y || true
  sudo apt-get install -y git git-lfs wget curl tmux build-essential || true
else
  echo "→ 无 sudo，跳过 apt。确保 git/git-lfs/wget/curl/tmux 已存在。"
fi
command -v git-lfs >/dev/null 2>&1 && git lfs install || echo "→ git-lfs 未装(模型下载改用 huggingface-cli 亦可)"

# Anaconda（若无则装到 ~/anaconda3）
if [ ! -x "$HOME/anaconda3/bin/conda" ]; then
  echo "→ 安装 Anaconda 到 ~/anaconda3 ..."
  AN=Anaconda3-2024.10-1-Linux-x86_64.sh
  wget -q "https://repo.anaconda.com/archive/$AN" -O "/tmp/$AN"
  bash "/tmp/$AN" -b -p "$HOME/anaconda3"
  rm -f "/tmp/$AN"
else
  echo "✓ Anaconda 已存在: $($HOME/anaconda3/bin/conda --version)"
fi
"$HOME/anaconda3/bin/conda" init bash || true
echo "======== 01 完成（重开 shell 或 source ~/.bashrc 使 conda 生效）========"
