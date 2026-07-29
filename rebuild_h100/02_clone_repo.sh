#!/usr/bin/env bash
# 02 克隆项目仓库到 ~/arthas/mentalDistill
set -euo pipefail
echo "======== 02 克隆仓库 ========"
REPO="https://github.com/arthaschan/mentalDistill.git"
DEST="$HOME/arthas/mentalDistill"
mkdir -p "$HOME/arthas"
if [ -d "$DEST/.git" ]; then
  echo "✓ 仓库已存在，拉取最新"; git -C "$DEST" pull --ff-only || true
else
  git clone "$REPO" "$DEST"
fi
echo "→ 仓库位置: $DEST"
echo "→ 注意: 仓库只含代码+文档。模型(models/)和大数据不在库里，由 04 脚本单独下载。"
echo "======== 02 完成 ========"
