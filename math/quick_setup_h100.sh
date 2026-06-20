#!/bin/bash
# H100 快速部署脚本 - 根据 AI科研工作台部署指南 调整

set -e  # 任何错误就停止

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   H100 AI 科研工作台快速部署脚本${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Step 0: 初始化 Conda
echo -e "${YELLOW}[Step 0] 初始化 Conda 环境...${NC}"
source ~/anaconda3/etc/profile.d/conda.sh
echo -e "${GREEN}✓ Conda 已初始化${NC}"
echo ""

# Step 1: 检查基础依赖
echo -e "${YELLOW}[Step 1] 检查基础依赖...${NC}"
conda --version
git --version
echo -e "${GREEN}✓ 基础依赖检查完成${NC}"
echo ""

# Step 2: 创建 Hermes 环境
echo -e "${YELLOW}[Step 2] 创建 Hermes Conda 环境...${NC}"
if conda env list | grep -q hermes; then
    echo -e "${GREEN}✓ Hermes 环境已存在，跳过${NC}"
else
    conda create -n hermes python=3.11 -y
    conda activate hermes
    pip install git+https://github.com/NousResearch/hermes-agent.git
    echo -e "${GREEN}✓ Hermes 环境创建完成${NC}"
fi
echo ""

# Step 3: 创建 vLLM 环境
echo -e "${YELLOW}[Step 3] 创建 vLLM Conda 环境...${NC}"
if conda env list | grep -q vllm; then
    echo -e "${GREEN}✓ vLLM 环境已存在，跳过${NC}"
else
    conda create -n vllm python=3.11 -y
    conda activate vllm
    # 固定版本：vLLM 0.9.2 + torch 2.7.0(cu126) + transformers≥4.51
    # 这个组合兼容 CUDA 驱动 12.9 且支持 Qwen3 架构
    pip install "vllm==0.9.2" "torch==2.7.0" --index-url https://download.pytorch.org/whl/cu126
    pip install "transformers>=4.51.1" "xformers==0.0.30"
    echo -e "${GREEN}✓ vLLM 环境创建完成${NC}"
fi
echo ""

# Step 4: 创建 Lean 环境
echo -e "${YELLOW}[Step 4] 创建 Lean Conda 环境...${NC}"
if conda env list | grep -q lean; then
    echo -e "${GREEN}✓ Lean 环境已存在，跳过${NC}"
else
    conda create -n lean python=3.11 -c conda-forge -y
    conda run -n lean conda install -c conda-forge lean lake -y
    echo -e "${GREEN}✓ Lean 环境创建完成${NC}"
fi
echo ""

# Step 5: 创建快速启动脚本
echo -e "${YELLOW}[Step 5] 创建快速启动脚本...${NC}"
mkdir -p ~/scripts

# 环境诊断脚本
cat > ~/scripts/check_env.sh << 'CHECKENV_SCRIPT'
#!/bin/bash
# AI 科研工作台 · 环境诊断
# 用法: check_env              # 快速体检
#       check_env --diagnose   # 体检 + Opus AI 诊断
#       check_env --only leankit # 仅 Lean/mathlib

set +e
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; NC='\033[0m'
PASS=0; FAIL=0; LOG_DIR=~/env_check_logs
ERROR_CTX=""; DIAGNOSE=false

check_pass() { PASS=$((PASS+1)); echo -e "  ${GREEN}✓${NC} $1"; }
check_fail() { FAIL=$((FAIL+1)); echo -e "  ${RED}✗${NC} $1"; }

run_check() {
    local name="$1" log="$LOG_DIR/$2.log"; shift 2
    printf "  ${CYAN}…${NC} %-40s" "$name"
    if "$@" > "$log" 2>&1; then
        echo -e "\r  ${GREEN}✓${NC} %-40s" "$name"
        PASS=$((PASS+1)); return 0
    else
        echo -e "\r  ${RED}✗${NC} %-40s" "$name"
        FAIL=$((FAIL+1))
        ERROR_CTX="${ERROR_CTX}\n\n[${name}]\n$(tail -60 "$log" 2>/dev/null)"
        return 1
    fi
}

check_system() {
    echo -e "\n${BLUE}── 系统基础 ──${NC}"
    run_check "操作系统" "os" uname -a
    run_check "磁盘 ~/" "disk" bash -c "df -h ~ | tail -1"
    run_check "内存" "mem" bash -c "free -h | grep Mem"
}
check_gpu() {
    echo -e "\n${BLUE}── GPU/CUDA ──${NC}"
    run_check "nvidia-smi" "nvidia" nvidia-smi
    run_check "CUDA" "cuda" bash -c "nvidia-smi | grep 'CUDA'"
}
check_conda() {
    echo -e "\n${BLUE}── Conda 环境 ──${NC}"
    source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
    run_check "Conda" "conda" conda --version
    for e in hermes vllm lean; do
        run_check "env:$e" "env_$e" bash -c "conda env list|grep -q '^${e} '"
    done
}
check_hermes() {
    echo -e "\n${BLUE}── Hermes Agent ──${NC}"
    source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
    run_check "hermes --version" "hermes_ver" bash -c "conda run -n hermes hermes --version"
    run_check "hermes doctor" "hermes_doc" bash -c "conda run -n hermes hermes doctor 2>&1 | head -30"
    run_check "config.yaml" "cfg" bash -c "test -f ~/.hermes/config.yaml"
    run_check ".env" "envf" bash -c "test -f ~/.hermes/.env"
}
check_vllm() {
    echo -e "\n${BLUE}── vLLM ──${NC}"
    source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
    run_check "vLLM 包" "vllm_pkg" bash -c "conda run -n vllm pip show vllm"
    run_check "Qwen3-32B 模型" "model" bash -c "test -f ~/models/Qwen3-32B/config.json"
    run_check "服务运行中" "vllm_svc" bash -c "curl -s --max-time 5 http://localhost:8000/v1/models >/dev/null 2>&1"
}
check_leankit() {
    echo -e "\n${BLUE}── Lean 4 + Mathlib ──${NC}"
    source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
    run_check "lean --version" "lean_ver" bash -c "conda run -n lean lean --version"
    run_check "lake --version" "lake_ver" bash -c "conda run -n lean lake --version"

    echo -e "\n  ${CYAN}[Mathlib 编译测试]${NC}"
    local ML="$LOG_DIR/mathlib.log" TD=$(mktemp -d)
    {
        echo "=== Mathlib 编译测试 ($(date)) ==="
        source ~/anaconda3/etc/profile.d/conda.sh
        conda activate lean 2>/dev/null || true
        cd "$TD"
        lake init math_test 2>&1 && cd math_test
        cat > lakefile.toml << 'LAKEFILE'
name = "math_test"
version = "0.1.0"

[[require]]
name = "mathlib"
git = "https://github.com/leanprover-community/mathlib4.git"
rev = "v4.18.0"
LAKEFILE
        lake update 2>&1
        lake exe cache get 2>&1 || true
        lake build 2>&1
    } > "$ML" 2>&1
    local rc=$?; rm -rf "$TD"
    if [ $rc -eq 0 ]; then
        echo -e "  ${GREEN}✓${NC} Mathlib 编译测试通过"
        PASS=$((PASS+1))
    else
        echo -e "  ${RED}✗${NC} Mathlib 编译测试失败 [日志: $ML]"
        FAIL=$((FAIL+1))
        ERROR_CTX="${ERROR_CTX}\n\n[Mathlib 编译]\n$(tail -60 "$ML")"
    fi
}
check_network() {
    echo -e "\n${BLUE}── 网络 ──${NC}"
    run_check "HuggingFace" "net_hf" bash -c "curl -s --max-time 10 https://huggingface.co >/dev/null 2>&1"
    run_check "GitHub" "net_gh" bash -c "curl -s --max-time 10 https://github.com >/dev/null 2>&1"
}
opus_diagnose() {
    echo -e "\n${BLUE}══ Opus 4.8 AI 诊断 ══${NC}"
    source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
    if ! conda run -n hermes hermes --version &>/dev/null; then
        echo -e "${RED}Hermes 不可用，跳过${NC}"; return 1
    fi
    local sys="OS: $(uname -a)\nGPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)\n"
    if [ -z "$ERROR_CTX" ]; then
        echo -e "${GREEN}无失败项，跳过 AI 诊断${NC}"; return 0
    fi
    echo -e "${YELLOW}正在调用 Opus 诊断...${NC}"
    local prompt="诊断 H100 环境故障。\n${sys}\n## 错误日志\n${ERROR_CTX}\n\n请分析根因，给出可直接执行的 bash 修复命令。中文回答，简洁。"
    echo -e "$prompt" > "$LOG_DIR/opus_prompt.txt"
    conda run -n hermes bash -c "echo '$prompt' | hermes chat --model claude-opus-4-8 --max-turns 1" 2>/dev/null | tail -n +5 | tee "$LOG_DIR/opus_result.txt"
}

main() {
    rm -rf "$LOG_DIR"; mkdir -p "$LOG_DIR"
    echo -e "${BLUE}╔════════════════════════════════╗${NC}"
    echo -e "${BLUE}║  AI 科研工作台 · 环境诊断报告        ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════╝${NC}"
    echo "  时间: $(date '+%Y-%m-%d %H:%M:%S')  主机: $(hostname)"
    local only="${1#--only=}"; [ "$1" = "--only" ] && only="$2"
    if [ -n "$only" ]; then
        echo -e "  ${YELLOW}仅检查: $only${NC}"
        case "$only" in
            system) check_system ;; gpu) check_gpu ;;
            conda) check_conda ;; hermes) check_hermes ;;
            vllm) check_vllm ;; lean*|math*) check_leankit ;;
            net*) check_network ;; *) check_system; check_gpu; check_conda; check_hermes; check_vllm; check_leankit; check_network ;;
        esac
    else
        check_system; check_gpu; check_conda; check_hermes; check_vllm; check_leankit; check_network
    fi
    local t=$((PASS+FAIL))
    echo -e "\n${BLUE}════════════════════════════════${NC}"
    echo -e "  总计 $t 项 | ${GREEN}通过 $PASS${NC} | ${RED}失败 $FAIL${NC}"
    if [ $FAIL -eq 0 ]; then echo -e "  ${GREEN}✓ 全部通过！${NC}"
    else echo -e "  ${RED}✗ 日志: $LOG_DIR${NC}"; fi
    echo -e "${BLUE}════════════════════════════════${NC}"

    if [ "$1" = "--diagnose" ] || [ "$1" = "diagnose" ]; then
        opus_diagnose
    elif [ $FAIL -gt 0 ]; then
        echo -e "\n${YELLOW}提示: check_env --diagnose  让 Opus AI 自动诊断${NC}"
    fi
}
main "$@"
CHECKENV_SCRIPT
chmod +x ~/scripts/check_env.sh
echo -e "${GREEN}✓ 环境诊断脚本已创建: ~/scripts/check_env.sh${NC}"

# vLLM 启动脚本
cat > ~/scripts/start_vllm.sh << 'EOF'
#!/bin/bash
source /home/student/anaconda3/etc/profile.d/conda.sh
conda activate vllm

python -m vllm.entrypoints.openai.api_server \
  --model ~/models/Qwen3-32B \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768 \
  --port 8000 \
  --trust-remote-code
EOF
chmod +x ~/scripts/start_vllm.sh
echo -e "${GREEN}✓ vLLM 启动脚本已创建: ~/scripts/start_vllm.sh${NC}"

# Hermes 启动脚本（tmux + 日志）
cat > ~/scripts/start_hermes.sh << 'EOF'
#!/bin/bash
# 在 tmux 中启动 Hermes，所有输出写到日志文件
# 用法: start_hermes            # 新建 tmux session 并启动
#        start_hermes attach    # 重新连接到已有 session

LOG=~/hermes_$(date +%Y%m%d_%H%M%S).log
SESSION=hermes

if [ "$1" = "attach" ]; then
    tmux attach -t "$SESSION"
    exit
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "⚠️  tmux session [$SESSION] 已存在"
    echo "   重新连接: tmux attach -t $SESSION"
    echo "   或运行: start_hermes attach"
    exit 1
fi

echo "日志文件: $LOG"
echo "启动后按 Ctrl+B 再按 D 可断开（任务继续跑）"
echo "重新连接: tmux attach -t $SESSION"
echo ""

tmux new -s "$SESSION" -d
tmux send-keys -t "$SESSION" "script -a $LOG" C-m
tmux send-keys -t "$SESSION" "source ~/anaconda3/etc/profile.d/conda.sh && conda activate hermes" C-m
tmux send-keys -t "$SESSION" "hermes chat --model claude-opus-4-8" C-m
tmux attach -t "$SESSION"
EOF
chmod +x ~/scripts/start_hermes.sh
echo -e "${GREEN}✓ Hermes 启动脚本已创建: ~/scripts/start_hermes.sh${NC}"
echo ""

# 快速初始化脚本
cat > ~/scripts/init_research_env.sh << 'EOF'
#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh

hermes_env() {
    conda activate hermes
    echo "✓ Hermes 环境已激活"
}

vllm_env() {
    conda activate vllm
    echo "✓ vLLM 环境已激活"
}

lean_env() {
    conda activate lean
    echo "✓ Lean 环境已激活"
}

start_vllm() {
    if pgrep -f "vllm.entrypoints.openai.api_server" > /dev/null; then
        echo "✓ vLLM 已在运行"
        curl -s http://localhost:8000/v1/models 2>/dev/null | head -5
    else
        echo "正在启动 vLLM..."
        nohup ~/scripts/start_vllm.sh > ~/vllm.log 2>&1 &
        echo $! > ~/vllm.pid
        sleep 5
        echo "✓ vLLM 启动中..."
    fi
}

stop_vllm() {
    if [ -f ~/vllm.pid ]; then
        kill $(cat ~/vllm.pid) 2>/dev/null
        rm -f ~/vllm.pid
        echo "✓ vLLM 已停止"
    else
        pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
        echo "✓ vLLM 进程已终止"
    fi
}

vllm_log() {
    tail -50 ~/vllm.log
}

start_hermes() {
    ~/scripts/start_hermes.sh
}

check_env() {
    ~/scripts/check_env.sh "$@"
}

echo -e "\n已加载研究工作台快捷命令:"
echo "  hermes_env      - 激活 Hermes 环境"
echo "  vllm_env        - 激活 vLLM 环境"
echo "  lean_env        - 激活 Lean 环境"
echo "  start_vllm      - 启动 vLLM 服务（后台）"
echo "  stop_vllm       - 停止 vLLM 服务"
echo "  vllm_log        - 查看 vLLM 日志"
echo "  start_hermes    - 在 tmux 中启动 Hermes（带日志，推荐）"
echo "  check_env        - 环境诊断"
echo "  check_env --diagnose - Opus AI 诊断"
echo "  test_all        - 测试所有服务\n"
EOF
chmod +x ~/scripts/init_research_env.sh
echo -e "${GREEN}✓ 快速初始化脚本已创建: ~/scripts/init_research_env.sh${NC}"
echo ""

# Step 6: 加载快捷命令
echo -e "${YELLOW}[Step 6] 加载快捷命令到 ~/.bashrc...${NC}"
if ! grep -q "init_research_env.sh" ~/.bashrc; then
    echo "source ~/scripts/init_research_env.sh" >> ~/.bashrc
    echo -e "${GREEN}✓ 已添加到 ~/.bashrc（下次登录生效）${NC}"
else
    echo -e "${GREEN}✓ 已在 ~/.bashrc 中（无需重复添加）${NC}"
fi
source ~/scripts/init_research_env.sh
echo ""

# 最后总结
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}部署完成！${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${YELLOW}后续步骤：${NC}"
echo "1. 下载 Qwen3-32B 模型到 ~/models/（约 65GB，首次需 1-2 小时）"
echo "   conda activate vllm"
echo "   huggingface-cli download Qwen/Qwen3-32B --local-dir ~/models/Qwen3-32B"
echo ""
echo "2. 配置灵眸AI API Key（模型名用完整名，不用短名）"
echo "   编辑 ~/.hermes/config.yaml 和 ~/.hermes/.env"
echo "   灵眸模型名格式: claude-opus-4-8（不是 opus）"
echo ""
echo "3. 启动 vLLM 服务（后台运行）"
echo "   nohup ~/scripts/start_vllm.sh > ~/vllm.log 2>&1 &"
echo "   或: start_vllm"
echo ""
echo "4. 启动 Hermes + Opus（推荐用 tmux，断开后继续跑）"
echo "   start_hermes              # 新建 tmux session 并启动"
echo "   start_hermes attach      # 重新连接已有 session"
echo ""
echo "5. 环境诊断"
echo "   check_env              (快速体检)"
echo "   check_env --diagnose   (体检 + Opus AI 诊断)"
echo "   check_env --only leankit  (仅检查 Lean/mathlib)"
echo ""
echo -e "${BLUE}═══ tmux 后台运行指南 ═══${NC}"
echo "  问题：SSH/VS Code 断开后 Hermes 进程会终止"
echo "  解决：用 tmux 让任务在后台跑，断开后继续"
echo ""
echo "  启动 Hermes（推荐方式）:"
echo "    start_hermes              # 新建 tmux session 并启动"
echo "    start_hermes attach      # 重新连接已有 session"
echo ""
echo "  手动 tmux 操作:"
echo "    tmux new -s hermes     # 新建 session"
echo "    Ctrl+B 再按 D         # 断开（任务继续跑）"
echo "    tmux attach -t hermes  # 重新连接"
echo ""
echo "  vLLM 后台运行:"
echo "    nohup ~/scripts/start_vllm.sh > ~/vllm.log 2>&1 &"
echo "    tail -f ~/vllm.log    # 实时看日志"
echo ""
echo -e "${YELLOW}快捷命令已就绪，可直接在终端使用：${NC}"
echo "   hermes_env, vllm_env, lean_env"
echo "   start_vllm, stop_vllm, vllm_log"
echo "   start_hermes, check_env, check_env --diagnose"
echo "   test_all"
echo ""
echo -e "${BLUE}更多信息见：AI科研工作台部署指南.md${NC}"
