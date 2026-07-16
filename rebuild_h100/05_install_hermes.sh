#!/usr/bin/env bash
# 05 安装 Hermes（uv + venv，Python 3.11）。装到 ~/.hermes/hermes-agent。
set -euo pipefail
echo "======== 05 安装 Hermes ========"
HDIR="$HOME/.hermes/hermes-agent"
REPO_SSH="git@github.com:NousResearch/hermes-agent.git"
REPO_HTTPS="https://github.com/NousResearch/hermes-agent.git"

# 1) 装 uv（Hermes 用它建 venv）
if ! command -v uv >/dev/null 2>&1; then
  echo "→ 安装 uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version || { echo "✗ uv 未装成功，检查 PATH 是否含 ~/.local/bin"; exit 1; }

# 2) clone hermes-agent
mkdir -p "$HOME/.hermes"
if [ -d "$HDIR/.git" ]; then
  echo "✓ hermes-agent 已存在，拉取最新"; git -C "$HDIR" pull --ff-only || true
else
  git clone "$REPO_HTTPS" "$HDIR" || git clone "$REPO_SSH" "$HDIR"
fi

# 3) 官方 setup 脚本（会用 uv 建 venv、装依赖）
cd "$HDIR"
if [ -f setup-hermes.sh ]; then
  bash setup-hermes.sh || echo "→ setup-hermes.sh 报错，尝试手动 venv"
fi
# 兜底：手动建 venv 装
if [ ! -x "$HDIR/venv/bin/hermes" ]; then
  uv venv --python 3.11 "$HDIR/venv"
  "$HDIR/venv/bin/python" -m pip install -e "$HDIR" || uv pip install --python "$HDIR/venv/bin/python" -e "$HDIR"
fi

# 4) 软链到 ~/.local/bin/hermes
mkdir -p "$HOME/.local/bin"
cat > "$HOME/.local/bin/hermes" <<'LAUNCH'
#!/usr/bin/env bash
unset PYTHONPATH
unset PYTHONHOME
exec "$HOME/.hermes/hermes-agent/venv/bin/hermes" "$@"
LAUNCH
sed -i "s|\$HOME|$HOME|g" "$HOME/.local/bin/hermes"
chmod +x "$HOME/.local/bin/hermes"

echo "======== 05 完成 ========"
echo "验证: hermes --version   （若命令找不到，把 ~/.local/bin 加入 PATH）"
echo "首次使用: hermes setup   配置 provider/model/API key（见 https://hermes-agent.nousresearch.com/docs）"
