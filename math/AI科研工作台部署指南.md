# AI 科研工作台部署指南

> **目标**：在港澳 H100 服务器上部署 Hermes Agent + vLLM + Lean 4，统一驱动 Claude Opus 4.8 / Qwen3-235B / Lean 形式化验证，支撑 AI+数学科研论文写作。
>
> **环境**：H100 80GB，位于港澳，可直连 Anthropic 官方 API。付费方式：灵眸AI 中转（微信充值，¥2.4/$）。

---

## 架构总览

```
┌─────────────────────────────────────────────────┐
│                Hermes Agent (主控)                │
│      任务编排 / 记忆 / 技能学习 / 工具调用         │
└──────────┬──────────────┬──────────┬─────────────┘
           │              │          │
    ┌──────▼──────┐ ┌─────▼─────┐ ┌──▼───────────┐
    │ Opus 4.8    │ │ Qwen3     │ │ Lean 4       │
    │ (灵眸AI中转) │ │ (H100本地) │ │ (形式化验证)  │
    │ 核心推理     │ │ 日常辅助   │ │ 数学证明     │
    └─────────────┘ └───────────┘ └─────────────┘
```

## 模型分工

| 角色 | 模型 | 部署方式 | 用途 | 预估月费 |
|------|------|---------|------|---------|
| **主力** | Claude Opus 4.8 | 灵眸AI 中转 | 数学推导、论文定稿、英文写作 | ~¥50-100/月 |
| **日常** | Qwen3-235B-A22B | H100 本地 vLLM | 读论文、LaTeX 格式化、代码草稿 | 免费 |
| **快速** | Qwen3-235B (辅助) | H100 本地 vLLM | 压缩、标题、审批分类 | 免费 |
| **验证** | Lean 4 + Mathlib | 本地安装 | 形式化证明验证 | 免费 |

---

## 第一部分：购买 Claude Opus 4.8 服务

### 灵眸AI 中转（微信充值，¥2.4/$，官价约 1/3）

**步骤**：

1. 访问灵眸AI 官网注册账号
2. 微信充值（建议先充 ¥50 试用）
3. 获取 API Key：`sk-xxx`
4. 记录 base_url：灵眸AI 提供的中转端点地址（后续配置需要）

**价格**（Opus 4.8，内部汇率 ¥2.4/$）：

| 项目 | 价格 |
|------|------|
| 输入 | ¥12 / 百万 token |
| 输出 | ¥60 / 百万 token |
| 缓存读取 | ¥1.20 / 百万 token |
| 加权均价 | ¥45.6 / 百万 token |

**成本估算**：
- 写一篇论文关键推导（~50K tokens）：¥5-10
- 月度科研预算：¥50-100
- 非核心任务用本地 Qwen3-235B，进一步降低成本

> **⚠️ 避坑**：不要大额充值（用多少充多少），灵眸AI 仅支持微信支付。

---

## 第二部分：部署 Hermes Agent

### Step 1：系统基础（H100 Anaconda 版本）

```bash
# SSH 登录 H100 服务器（如未登录）
ssh user@your-h100-server

# 检查 Anaconda 是否已安装
conda --version
# 如果提示 conda: command not found，需要初始化 conda
source ~/anaconda3/etc/profile.d/conda.sh

# 检查已有的系统依赖（git、ffmpeg 等）
git --version
which ffmpeg
which docker
which node

# 如果缺少某些依赖，可选择安装：
# - ripgrep: conda install -c conda-forge ripgrep
# - ffmpeg: conda install -c conda-forge ffmpeg
# - docker: 需要 sudo，或联系管理员安装
# - node: conda install nodejs (推荐用 conda 而非系统包)

# ⚠️ 公共服务器：不要 sudo 修改系统包或 Systemd 服务
```

### Step 2：创建 Hermes Conda 环境并安装

```bash
# 初始化 conda（如未初始化）
source ~/anaconda3/etc/profile.d/conda.sh

# 为 Hermes 创建专属环境（Node.js 22.x + Python）
conda create -n hermes -c conda-forge nodejs=22 python=3.11 -y
conda activate hermes

# 用 npm 安装 Hermes（官方推荐）
npm install -g @hermes-ai/cli

# 或从源代码安装（如需要最新开发版本）
git clone https://github.com/NousResearch/hermes-agent.git ~/hermes-source
cd ~/hermes-source
npm install
npm link

# 验证安装
hermes --version
hermes doctor
```

### Step 3：配置 config.yaml（Hermes 环境激活下）

创建配置文件：

```bash
# 确保 Hermes 环境激活
conda activate hermes

mkdir -p ~/.hermes
cat > ~/.hermes/config.yaml << 'HERMES_CONFIG'
# ===== 自定义 Provider =====
custom_providers:
  # 灵眸AI 中转（主力模型 Opus 4.8）
  - id: "anthropic-relay"
    type: "anthropic"
    base_url: "https://灵眸AI的中转地址/anthropic"
    api_key: "${ANTHROPIC_API_KEY}"
    models:
      - "claude-opus-4-8-20250527"
      - "claude-sonnet-4-5-20250514"
    extra_body: {}

  # 本地 vLLM（日常模型）
  - id: "vllm-local"
    type: "openai-compat"
    base_url: "http://localhost:8000/v1"
    api_key: "EMPTY"
    models:
      - "Qwen/Qwen3-235B-A22B"
    timeout: 300

# ===== 主模型（核心推理用 Opus 4.8）=====
model:
  default: "claude-opus-4-8-20250527"
  provider: "anthropic-relay"
  reasoning_effort: "high"

# ===== 辅助模型（用本地免费模型省钱）=====
auxiliary:
  compression:
    provider: "vllm-local"
    model: "Qwen/Qwen3-235B-A22B"
  vision:
    provider: "vllm-local"
    model: "Qwen/Qwen3-235B-A22B"
  web_extract:
    provider: "vllm-local"
    model: "Qwen/Qwen3-235B-A22B"
  title_generation:
    provider: "vllm-local"
    model: "Qwen/Qwen3-235B-A22B"
  approval:
    provider: "vllm-local"
    model: "Qwen/Qwen3-235B-A22B"
  skills_hub:
    provider: "vllm-local"
    model: "Qwen/Qwen3-235B-A22B"

# ===== 模型别名（快速切换）=====
model_aliases:
  opus: "claude-opus-4-8-20250527"
  sonnet: "claude-sonnet-4-5-20250514"
  qwen: "Qwen/Qwen3-235B-A22B"

# ===== Provider 超时配置 =====
providers:
  anthropic-relay:
    request_timeout_seconds: 1800
  vllm-local:
    request_timeout_seconds: 300

# ===== 终端配置 =====
terminal:
  backend: "docker"
  timeout: 180
  container_persistent: true

# ===== Agent 行为 =====
agent:
  max_turns: 90
  api_max_retries: 3

# ===== 记忆系统 =====
memory:
  memory_enabled: true
  user_profile_enabled: true
  memory_char_limit: 2200

# ===== 审批模式 =====
approvals:
  mode: "smart"

# ===== 显示 =====
display:
  language: "zh"
  show_cost: true
HERMES_CONFIG
```

> **⚠️ 重要**：将 `base_url` 替换为灵眸AI 实际提供的中转地址。

### Step 4：配置 .env（Hermes 环境激活下）

```bash
# 在 Hermes 环境中创建配置
conda activate hermes

cat > ~/.hermes/.env << 'HERMES_ENV'
# 灵眸AI API Key（主力模型，微信充值获取）
ANTHROPIC_API_KEY=sk-替换为灵眸AI的Key

# vLLM 本地模型（辅助）
OPENAI_API_KEY=EMPTY
OPENAI_BASE_URL=http://localhost:8000/v1
HERMES_ENV
```

> **⚠️ 重要**：将 `sk-替换为灵眸AI的Key` 替换为你的实际 Key。

### Step 5：启用工具

```bash
hermes tools enable file shell browser search
```

### Step 6：验证

```bash
hermes config check
hermes doctor
# 测试 Opus 4.8（需要灵眸AI Key 配置好）
hermes chat --model opus
# 测试本地 Qwen（需要 vLLM 启动后，见第三部分）
hermes chat --model qwen
```

---

## 第三部分：部署本地 vLLM + Qwen3-235B（Anaconda 版本）

### Step 1：创建 vLLM Conda 环境

```bash
# 初始化 conda（如未初始化）
source ~/anaconda3/etc/profile.d/conda.sh

# 为 vLLM 创建专属环境
conda create -n vllm python=3.11 -y
conda activate vllm

# 安装 vLLM（H100 + CUDA 的优化版本）
pip install vllm torch
```

### Step 2：下载模型

```bash
# 安装 HuggingFace 工具
pip install huggingface_hub

# 从 HuggingFace 下载（首次需要几小时，模型约 150GB）
huggingface-cli download Qwen/Qwen3-235B-A22B \
  --local-dir ~/models/Qwen3-235B-A22B
```

> **💡 加速下载**：如果 HuggingFace 下载慢，可以用镜像站：
> ```bash
> export HF_ENDPOINT=https://hf-mirror.com
> ```

### Step 3：启动 vLLM 服务（Conda 版本）

```bash
# 创建启动脚本（推荐用脚本而非 systemd，便于公共服务器管理）
mkdir -p ~/scripts
cat > ~/scripts/start_vllm.sh << 'EOF'
#!/bin/bash
source /home/student/anaconda3/etc/profile.d/conda.sh
conda activate vllm

python -m vllm.entrypoints.openai.api_server \
  --model ~/models/Qwen3-235B-A22B \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 32768 \
  --port 8000 \
  --trust-remote-code
EOF

chmod +x ~/scripts/start_vllm.sh

# 后台启动 vLLM
nohup ~/scripts/start_vllm.sh > ~/vllm.log 2>&1 &
echo $! > ~/vllm.pid  # 保存 PID 以便后续关闭

# 查看日志确认启动成功
tail -f ~/vllm.log
# 看到 "Uvicorn running on http://0.0.0.0:8000" 即成功
```

### Step 4：验证 vLLM

```bash
# 测试本地模型
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-235B-A22B",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 50
  }'
```

### Step 5：管理 vLLM 进程（不使用 Systemd，保护公共服务器）

由于 H100 是公共服务器，**不使用 Systemd 全局服务**。改为用户级管理脚本：

```bash
# 关闭 vLLM
kill $(cat ~/vllm.pid 2>/dev/null) 2>/dev/null || echo "vLLM 未运行"
rm -f ~/vllm.pid

# 查看 vLLM 日志
tail -f ~/vllm.log

# 重启 vLLM
~/scripts/start_vllm.sh &
echo $! > ~/vllm.pid

# 检查 vLLM 是否运行
ps aux | grep vllm | grep -v grep
```

**如需开机自启**（需管理员权限），可联系服务器管理员配置 Systemd，或者用 `crontab -e` 个人定时任务：

```bash
# 添加到 crontab（@reboot 表示开机执行）
@reboot /home/student/scripts/start_vllm.sh >> /home/student/vllm.log 2>&1
```

---

## 第四部分：添加数学证明工具链（替代 Gauss / AlphaProof Nexus）

### 为什么不用 Gauss / AlphaProof Nexus？

| 工具 | 现状 |
|------|------|
| **Google Gauss**（DeepMind 数学智能体） | 仅面向受邀职业数学家有限测试，未公开 API |
| **AlphaProof Nexus** | 仅开源了 9 道 Erdős 问题的 Lean 证明代码，系统本身未开源 |

**替代方案**：Lean 4 形式化数学工具链，可在本地部署，实现类似的形式化验证能力。

### Step 1：安装 Lean 4（使用 Conda 或官方方式）

**方式 A：使用 Conda（推荐，最简单）**

```bash
# 创建 Lean 专属环境
conda create -n lean python=3.11 -c conda-forge -y
conda activate lean

# 安装 Lean 4 和 Lake
conda install -c conda-forge lean lake -y

# 验证
lean --version
lake --version
```

**方式 B：官方安装（如需最新版本）**

```bash
# 安装 Elan（Lean 版本管理器）
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh

# 初始化 shell（选择 bash）
source ~/.elan/env

# 验证
lean --version
lake --version
```

### Step 2：安装 Mathlib（Lean 环境激活下）

```bash
# 激活 Lean 环境（如使用 Conda）
conda activate lean

# 创建项目并依赖 Mathlib
mkdir -p ~/math-research && cd ~/math-research
lake init math_research

# 配置 lakefile.lean 添加 Mathlib 依赖
cat > lakefile.lean << 'EOF'
import Lake
open Lake DSL

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git"

@[default_target]
lean_lib MathResearch where
EOF

# 获取缓存（加速编译）
lake exe cache get

# 编译（首次编译较慢，约 5-15 分钟）
lake build
```

### Step 3：安装 LeanDojo（机器学习辅助证明搜索）

```bash
pip install lean-dojo
```

### Step 4：创建 Hermes Lean 验证技能

在 `~/.hermes/skills/` 下创建技能文件，让 Hermes 能自动调用 Lean 验证：

```bash
mkdir -p ~/.hermes/skills/lean-verify
cat > ~/.hermes/skills/lean-verify/SKILL.md << 'EOF'
# Lean 4 形式化验证技能

## 用途
将数学推导转换为 Lean 4 代码并用编译器验证正确性。

## 触发条件
当用户要求验证数学证明、形式化推导、或使用 Lean 4 时触发。

## 执行步骤
1. 将用户的自然语言数学推导转换为 Lean 4 代码
2. 将代码写入 ~/math-research/MathResearch/ 验证文件
3. 执行 `lake build` 验证
4. 如果验证失败，分析编译器错误信息并修正代码
5. 循环执行直到验证通过或达到最大重试次数（10次）

## 可用命令
- `lake build`：编译验证
- `lean ~/math-research/MathResearch/Theorem.lean`：单文件验证
EOF
```

### Hermes 驱动的数学工作流

```
用户提出数学问题
    ↓
Hermes → Opus 4.8：自然语言推导和证明
    ↓
Hermes → Qwen3-235B：将推导转为 Lean 4 代码草稿
    ↓
Hermes → Lean 4 编译器：验证证明
    ↓ (失败则反馈错误)
Hermes → Opus 4.8：根据错误修正证明
    ↓ (循环直到通过)
输出：经过形式化验证的数学证明
```

---

## 第五部分：日常使用

### 快速切换模型

```bash
# 核心推导 → Opus 4.8（付费，最强）
hermes chat --model opus

# 日常问答 → 本地 Qwen（免费）
hermes chat --model qwen

# 论文润色 → Sonnet（便宜）
hermes chat --model sonnet
```

### 科研场景示例

```bash
# 1. 读论文总结（本地 Qwen，免费）
hermes "读 ~/papers/attention-is-all-you-need.pdf，用中文总结核心贡献"

# 2. 数学推导（Opus 4.8，付费但最强）
hermes --model opus "推导 Doubly Robust 估计量的渐近正态性"

# 3. 写 LaTeX 论文（Opus 起草 + Qwen 格式化）
hermes --model opus "帮我写论文 Method 部分的因果效应估计框架"
hermes "把上面的内容格式化为 ICML 2026 LaTeX 模板"

# 4. 形式化验证（Lean 4）
hermes "把刚才的推导转换为 Lean 4 代码并用 lake build 验证"

# 5. 实验代码
hermes --model qwen "写 PyTorch 脚本实现 IPW 因果效应估计"

# 6. 批量处理
hermes "读 ~/papers/ 目录下所有 PDF，生成每篇的中文摘要，保存到 summaries.md"
```

---

## 实施检查清单

| 序号 | 任务 | 预估时间 | 状态 |
|:---:|------|:---:|:---:|
| 1 | 注册灵眸AI + 微信充值 ¥50 获取 API Key | 30 分钟 | ☐ |
| 2 | 检查 Anaconda 已安装在 `/home/student/anaconda3` | 5 分钟 | ☐ |
| 3 | 创建 Hermes Conda 环境并安装 | 20 分钟 | ☐ |
| 4 | 配置 Hermes config.yaml 和 .env | 15 分钟 | ☐ |
| 5 | 创建 vLLM Conda 环境 | 10 分钟 | ☐ |
| 6 | 下载 Qwen3-235B 模型到 ~/models/ | 3-6 小时 | ☐ |
| 7 | 启动 vLLM 脚本 + 验证连接 | 10 分钟 | ☐ |
| 8 | 创建 Lean Conda 环境并安装 | 15 分钟 | ☐ |
| 9 | 创建 Lean 项目 + Mathlib 初始编译 | 20 分钟 | ☐ |
| 10 | 创建 Lean 验证技能 | 30 分钟 | ☐ |
| 11 | 端到端测试：Opus 推导 → Qwen 格式化 → Lean 验证 | 15 分钟 | ☐ |

---

## 附录 A：H100 快速初始化脚本

如果每次都手动激活环境很麻烦，可以创建快捷脚本：

```bash
# 创建快速初始化脚本
cat > ~/scripts/init_research_env.sh << 'INIT_EOF'
#!/bin/bash
# 初始化研究工作台环境

# 初始化 Conda
source ~/anaconda3/etc/profile.d/conda.sh

# 函数：激活 Hermes 环境
hermes_env() {
    conda activate hermes
    echo "✓ Hermes 环境已激活"
}

# 函数：激活 vLLM 环境
vllm_env() {
    conda activate vllm
    echo "✓ vLLM 环境已激活"
}

# 函数：激活 Lean 环境
lean_env() {
    conda activate lean
    echo "✓ Lean 环境已激活"
}

# 函数：启动 vLLM 服务
start_vllm() {
    if pgrep -f "vllm.entrypoints.openai.api_server" > /dev/null; then
        echo "✓ vLLM 已在运行"
        curl http://localhost:8000/v1/models 2>/dev/null | head -20
    else
        echo "正在启动 vLLM..."
        ~/scripts/start_vllm.sh &
        echo $! > ~/vllm.pid
        sleep 5
        echo "✓ vLLM 启动中，10 秒后再试..."
    fi
}

# 函数：停止 vLLM 服务
stop_vllm() {
    if [ -f ~/vllm.pid ]; then
        kill $(cat ~/vllm.pid) 2>/dev/null
        rm -f ~/vllm.pid
        echo "✓ vLLM 已停止"
    else
        pkill -f "vllm.entrypoints.openai.api_server"
        echo "✓ vLLM 进程已终止"
    fi
}

# 函数：查看 vLLM 日志
vllm_log() {
    tail -50 ~/vllm.log
}

# 函数：快速测试所有服务
test_all() {
    echo "=== 测试 vLLM ==="
    curl -s http://localhost:8000/v1/models | jq .
    
    echo ""
    echo "=== 测试 Hermes ==="
    hermes_env
    hermes --version
    hermes doctor
}

echo "已加载研究工作台快捷命令:"
echo "  hermes_env      - 激活 Hermes 环境"
echo "  vllm_env        - 激活 vLLM 环境"
echo "  lean_env        - 激活 Lean 环境"
echo "  start_vllm      - 启动 vLLM 服务"
echo "  stop_vllm       - 停止 vLLM 服务"
echo "  vllm_log        - 查看 vLLM 日志"
echo "  test_all        - 测试所有服务"
INIT_EOF

chmod +x ~/scripts/init_research_env.sh

# 加载到 ~/.bashrc（一次性）
echo "source ~/scripts/init_research_env.sh" >> ~/.bashrc

# 立即加载
source ~/scripts/init_research_env.sh
```

然后每次登录或新开终端时，直接用快捷命令：

```bash
# 激活 Hermes
hermes_env

# 启动 vLLM
start_vllm

# 查看 vLLM 日志
vllm_log

# 停止 vLLM
stop_vllm
```

---

## 附录 B：Conda 环境信息查询

```bash
# 列出所有环境
conda env list

# 查看当前环境的包
conda list

# 查看特定环境的包
conda list -n hermes

# 导出环境配置（备份用）
conda env export -n hermes > hermes_env.yml

# 从备份恢复环境
conda env create -f hermes_env.yml
```

---

---

## 附录 C：H100 已安装依赖检查

根据本次检查（2026-06-15）：

| 依赖 | 状态 | 位置/版本 | 备注 |
|------|------|---------|------|
| **Anaconda** | ✓ | `/home/student/anaconda3` conda 25.5.1 | 完全可用 |
| **git** | ✓ | 2.34.1 | 完全可用 |
| **Python 3** | ✓ | 系统自带 | conda 会用自己的版本 |
| **ripgrep** | ✗ | - | 可选，用 `conda install -c conda-forge ripgrep` |
| **ffmpeg** | ✗ | - | 可选，用 `conda install -c conda-forge ffmpeg` |
| **Docker** | ✗ | - | 可选（Hermes 终端沙箱），需管理员权限 |
| **Node.js** | ✗ | - | 必需，用 `conda install nodejs` 装 |
| **vLLM** | - | - | 在 vllm 环境中安装 |
| **Lean 4** | - | - | 在 lean 环境中安装（conda 或官方） |

**可选依赖的安装**（如需要）：

```bash
# 如果需要 Hermes 终端沙箱和高级文件搜索
conda install -c conda-forge ripgrep ffmpeg -y

# 如果需要使用 Docker 沙箱（需 sudo，不推荐在公共服务器）
# 联系管理员安装 docker
```

---

## 常见问题
```bash
# 检查 GPU 状态
nvidia-smi
# 确保 CUDA 版本兼容
nvcc --version
```

### Q: 灵眸AI 连接超时
```bash
# 从服务器测试连通性
curl -v https://灵眸AI的中转地址
# 如果超时，可能需要配置代理
export HTTPS_PROXY=http://your-proxy:port
```

### Q: Hermes 找不到模型
```bash
# 检查配置
hermes model
# 确认 vLLM 服务在运行
curl http://localhost:8000/v1/models
```

### Q: H100 显存不够跑 Qwen3-235B？
- Qwen3-235B-A22B 是 MoE 架构，激活参数仅 22B，H100 80GB 完全够用
- 如果不行，改用 Qwen3-32B（约 35GB 显存）

---

## 附录 D：H100 公共服务器部署注意事项

本文档已针对 H100 的 Anaconda 环境进行调整。关键变更：

| 项 | Mac 原版 | H100 调整版 | 原因 |
|---|---------|-----------|------|
| **Python 环境** | `python3 -m venv` | `conda create -n env` | Anaconda 环境管理更方便 |
| **Hermes 安装** | 全局脚本 | conda hermes 环境 | 隔离环境，不污染全局 |
| **vLLM 启动** | Systemd 服务 | nohup 脚本 + PID 管理 | 公共服务器，不能用 Systemd |
| **日志管理** | systemctl status | nohup + tail | 用户级管理，无需 sudo |
| **环境初始化** | 手动 source | ~/.bashrc 快捷脚本 | 提升开发体验 |

**公共服务器最佳实践**：

1. ✓ **每个项目用独立的 Conda 环境** → 避免版本冲突
2. ✓ **用 nohup 启动长时间任务** → 断开连接也不会中断
3. ✗ **不要用 Systemd/cron** → 除非有管理员权限
4. ✓ **定期检查 PID 和日志** → 发现僵尸进程
5. ✓ **留下清晰的启停文档** → 便于他人接手

**vLLM 进程管理**：

```bash
# 查看是否运行
ps aux | grep vllm | grep -v grep

# 查看端口占用
lsof -i :8000

# 强制停止（如需要）
pkill -9 -f vllm.entrypoints.openai.api_server

# 定期清理日志（可选）
tail -1000 ~/vllm.log > ~/vllm.log.tmp && mv ~/vllm.log.tmp ~/vllm.log
```
