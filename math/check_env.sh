#!/bin/bash
# ============================================================
# AI 科研工作台 — 环境诊断脚本
#   用法: check_env              # 快速体检
#         check_env --diagnose   # 体检 + Opus AI 诊断失败项
#         check_env --only leankit # 只检查 Lean/mathlib
# ============================================================

set +e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

PASS=0
FAIL=0
DIAGNOSE_MODE=false
LOG_DIR=~/env_check_logs
SUMMARY=""
ERROR_CONTEXTS=""

# ========================
# 工具函数
# ========================
init_log_dir() {
    rm -rf "$LOG_DIR"
    mkdir -p "$LOG_DIR"
}

check_pass() {
    local name="$1"
    local detail="${2:-}"
    PASS=$((PASS + 1))
    echo -e "  ${GREEN}✓${NC} $name ${detail}"
}

check_fail() {
    local name="$1"
    local detail="${2:-}"
    FAIL=$((FAIL + 1))
    echo -e "  ${RED}✗${NC} $name ${RED}$detail${NC}"
}

run_check() {
    local name="$1"
    local logfile="$LOG_DIR/${2}.log"
    shift 2
    local cmd=("$@")

    echo -n "  ${CYAN}…${NC} $name "

    if "${cmd[@]}" > "$logfile" 2>&1; then
        echo -e "\r  ${GREEN}✓${NC} $name"
        PASS=$((PASS + 1))
        return 0
    else
        echo -e "\r  ${RED}✗${NC} $name ${RED}[失败，日志: $logfile]${NC}"
        FAIL=$((FAIL + 1))
        ERROR_CONTEXTS="${ERROR_CONTEXTS}\n\n===== 检查项: $name =====\n$(cat "$logfile" 2>/dev/null | tail -80)"
        return 1
    fi
}

# ========================
# 各组件检查函数
# ========================

check_system() {
    echo ""
    echo -e "${BLUE}── 系统基础 ──${NC}"

    run_check "操作系统" "os" uname -a
    run_check "磁盘空间 ~/" "disk_home" bash -c "df -h ~ | tail -1 | awk '{print \$4\" 可用 / \"\$2\" 总量\"}'"
    run_check "磁盘空间 /tmp" "disk_tmp" bash -c "df -h /tmp | tail -1 | awk '{print \$4\" 可用\"}'"
    run_check "内存" "memory" bash -c "free -h | grep Mem | awk '{print \$7\" 可用 / \"\$2\" 总量\"}'"
}

check_gpu() {
    echo ""
    echo -e "${BLUE}── GPU / CUDA ──${NC}"

    run_check "nvidia-smi" "nvidia_smi" nvidia-smi
    run_check "CUDA 版本" "cuda_version" bash -c "nvcc --version 2>&1 | grep 'release' || nvidia-smi | grep 'CUDA'"
    run_check "GPU 型号" "gpu_model" bash -c "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"
}

check_conda() {
    echo ""
    echo -e "${BLUE}── Conda 环境 ──${NC}"

    source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null

    run_check "Conda 可用" "conda_version" conda --version

    for env in hermes vllm lean; do
        run_check "Conda 环境: $env" "conda_env_${env}" \
            bash -c "source ~/anaconda3/etc/profile.d/conda.sh && conda env list | grep -q '^${env} '"
    done
}

check_hermes() {
    echo ""
    echo -e "${BLUE}── Hermes Agent ──${NC}"

    source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null

    run_check "Hermes 版本" "hermes_version" \
        bash -c "source ~/anaconda3/etc/profile.d/conda.sh && conda activate hermes && hermes --version"

    run_check "Hermes doctor" "hermes_doctor" \
        bash -c "source ~/anaconda3/etc/profile.d/conda.sh && conda activate hermes && hermes doctor"

    run_check "config.yaml 存在" "hermes_config" \
        bash -c "test -f ~/.hermes/config.yaml"

    run_check ".env 存在" "hermes_env" \
        bash -c "test -f ~/.hermes/.env"
}

check_vllm() {
    echo ""
    echo -e "${BLUE}── vLLM 本地推理 ──${NC}"

    source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null

    run_check "vLLM 包已安装" "vllm_installed" \
        bash -c "source ~/anaconda3/etc/profile.d/conda.sh && conda run -n vllm pip show vllm"

    # 检查模型是否下载
    local model_path="$HOME/models/Qwen3-32B"
    run_check "Qwen3-32B 模型已下载" "model_exists" \
        bash -c "test -d $model_path && test -f $model_path/config.json"

    # 检查 vLLM 服务进程
    run_check "vLLM 服务运行中" "vllm_running" \
        bash -c "curl -s --max-time 5 http://localhost:8000/v1/models >/dev/null 2>&1"

    # 检查 GPU 显存占用（应该有进程在用）
    run_check "GPU 显存占用" "gpu_memory" \
        bash -c "nvidia-smi --query-gpu=memory.used --format=csv,noheader | head -1 | grep -v '0 MiB'"
}

check_leankit() {
    echo ""
    echo -e "${BLUE}── Lean 4 + Mathlib 形式化验证 ──${NC}"

    source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null

    # Lean 4 编译器
    run_check "Lean 4 编译器" "lean_version" \
        bash -c "source ~/anaconda3/etc/profile.d/conda.sh && conda run -n lean lean --version"

    # Lake 构建工具
    run_check "Lake 构建工具" "lake_version" \
        bash -c "source ~/anaconda3/etc/profile.d/conda.sh && conda run -n lean lake --version"

    # elan 版本管理器（如果安装了）
    run_check "elan 版本管理器" "elan_version" \
        bash -c "source ~/.elan/bin/lean-init 2>/dev/null && elan --version || which lean"

    # =====  Mathlib 完整测试 =====
    echo ""
    echo -e "  ${CYAN}[Mathlib 功能测试]${NC} 创建临时项目并编译..."

    local MATHLIB_LOG="$LOG_DIR/mathlib_build.log"
    local TEST_DIR=$(mktemp -d)

    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate lean 2>/dev/null || true

    {
        echo "=== Mathlib 编译测试 ==="
        echo "时间: $(date)"
        echo "测试目录: $TEST_DIR"
        echo ""

        cd "$TEST_DIR"

        # 尝试安装 mathlib4 缓存（使用 elan，如果 conda lean 不够用）
        if command -v lake &>/dev/null; then
            echo "lake 已找到: $(which lake)"
        else
            echo "ERROR: lake 命令找不到"
        fi

        # 创建测试项目（用 lakefile.toml 格式，Lean 4.18.0 默认）
        if command -v lake &>/dev/null; then
            lake init math_test 2>&1 || echo "lake init 失败（可能已有项目）"
            cd math_test

            # 写入 lakefile.toml 添加 mathlib 依赖
            cat > lakefile.toml << 'LAKEFILE'
name = "math_test"
version = "0.1.0"

[[require]]
name = "mathlib"
git = "https://github.com/leanprover-community/mathlib4.git"
rev = "v4.18.0"
LAKEFILE

            echo ""
            echo "=== lake update (拉取 mathlib) ==="
            lake update 2>&1
            echo ""
            echo "=== lake exe cache get (获取预编译缓存) ==="
            lake exe cache get 2>&1 || echo "(缓存获取失败，将尝试从源码编译)"
            echo ""
            echo "=== lake build (编译) ==="
            lake build 2>&1
        else
            echo "SKIP: lake 不可用，无法完成 mathlib 编译测试"
        fi

        echo ""
        echo "=== 测试完成 ==="
    } > "$MATHLIB_LOG" 2>&1

    local mathlib_rc=$?
    rm -rf "$TEST_DIR"

    if [ $mathlib_rc -eq 0 ]; then
        echo -e "  ${GREEN}✓${NC} Mathlib 编译测试 ${GREEN}通过${NC}"
        PASS=$((PASS + 1))
    else
        echo -e "  ${RED}✗${NC} Mathlib 编译测试 ${RED}失败${NC} [日志: $MATHLIB_LOG]"
        FAIL=$((FAIL + 1))
        ERROR_CONTEXTS="${ERROR_CONTEXTS}\n\n===== 检查项: Mathlib 编译测试 =====\n$(tail -80 "$MATHLIB_LOG" 2>/dev/null)"
    fi
}

check_network() {
    echo ""
    echo -e "${BLUE}── 网络连通性 ──${NC}"

    run_check "HuggingFace" "net_hf" \
        bash -c "curl -s --max-time 10 https://huggingface.co >/dev/null 2>&1"

    run_check "GitHub" "net_gh" \
        bash -c "curl -s --max-time 10 https://github.com >/dev/null 2>&1"

    run_check "Google" "net_google" \
        bash -c "curl -s --max-time 10 https://www.google.com >/dev/null 2>&1"
}

# ========================
# Opus AI 诊断
# ========================

run_opus_diagnose() {
    echo ""
    echo -e "${BLUE}══════════════════════════════════════${NC}"
    echo -e "${BLUE}  Opus 4.8 AI 诊断模式${NC}"
    echo -e "${BLUE}══════════════════════════════════════${NC}"
    echo ""

    source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null

    # 检查 hermes 是否能正常工作
    if ! conda run -n hermes hermes --version &>/dev/null; then
        echo -e "${RED}Hermes 不可用，跳过 AI 诊断${NC}"
        echo "请先安装 Hermes 并配置 Opus API Key"
        return 1
    fi

    # 收集系统信息
    local sys_info=""
    sys_info+="操作系统: $(uname -a 2>/dev/null)\n"
    sys_info+="GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo '未知')\n"
    sys_info+="Conda: $(conda --version 2>/dev/null || echo '未安装')\n"
    sys_info+="$(free -h 2>/dev/null | head -2)\n"

    echo -e "${YELLOW}收集错误上下文...${NC}"

    # 如果没有明确错误，扫描日志文件
    if [ -z "$ERROR_CONTEXTS" ]; then
        ERROR_CONTEXTS="（无显式检查失败项）\n\n请分析以下日志中是否有潜在问题：\n"
        for logfile in "$LOG_DIR"/*.log; do
            if [ -s "$logfile" ]; then
                local fname=$(basename "$logfile")
                local err_lines=$(grep -i -E "error|fail|warning|cannot|not found|timeout|拒绝|cancel" "$logfile" 2>/dev/null | head -20)
                if [ -n "$err_lines" ]; then
                    ERROR_CONTEXTS="${ERROR_CONTEXTS}\n--- ${fname} 中发现可疑行 ---\n${err_lines}\n"
                fi
            fi
        done
    fi

    # 构建诊断 prompt
    local prompt=$(cat <<PROMPT_END
你是 AI 科研工作台的运维助手。请诊断以下 H100 服务器环境部署问题。

## 系统信息
${sys_info}

## 错误日志
${ERROR_CONTEXTS}

## 诊断要求
1. 分析每个错误的根因（是网络问题、版本不兼容、依赖缺失还是配置错误）
2. 按优先级排列修复建议（最可能解决的一步放前面）
3. 每条修复建议给出可直接执行的 bash 命令
4. 特别关注 Lean 4 + mathlib4 安装失败问题（常见原因：elan 版本、GitHub 连通性、缓存获取失败）

请用中文简洁回答，每个问题 3-5 行。
PROMPT_END
)

    echo -e "${YELLOW}正在调用 Opus 4.8 诊断...${NC}（可能需要 30-60 秒）"
    echo ""

    # 保存 prompt 供调试
    echo -e "$prompt" > "$LOG_DIR/opus_diagnose_prompt.txt"

    # 调用 Hermes + Opus（用完整模型名，短别名 opus 在 v0.16.0 不生效）
    conda run -n hermes bash -c "
        echo '$prompt' | hermes chat --model claude-opus-4-8 --max-turns 1 2>/dev/null
    " 2>&1 | tail -n +5 | tee "$LOG_DIR/opus_diagnose_result.txt"

    local rc=${PIPESTATUS[0]}
    if [ $rc -eq 0 ]; then
        echo ""
        echo -e "${GREEN}Opus 诊断完成，完整结果已保存到: $LOG_DIR/opus_diagnose_result.txt${NC}"
    else
        echo -e "${RED}Opus 诊断调用失败 (rc=$rc)，检查 Hermes 配置${NC}"
    fi
}

# ========================
# 主流程
# ========================

main() {
    local mode="${1:-}"
    local only="${2:-}"

    # 解析 --only 参数
    case "$mode" in
        --only)
            only="$2"
            mode="check"
            ;;
        --diagnose)
            DIAGNOSE_MODE=true
            ;;
    esac
    # 兼容: check_env --diagnose 和 check_env diagnose
    if [ "$1" = "diagnose" ]; then
        DIAGNOSE_MODE=true
    fi
    if [ "$1" = "--diagnose" ]; then
        DIAGNOSE_MODE=true
    fi

    init_log_dir

    echo -e "${BLUE}╔══════════════════════════════════╗${NC}"
    echo -e "${BLUE}║   AI 科研工作台 · 环境诊断报告        ║${NC}"
    echo -e "${BLUE}╚══════════════════════════════════╝${NC}"
    echo -e "  时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo -e "  主机: $(hostname)"
    echo -e "  日志: $LOG_DIR"
    echo ""

    if [ -n "$only" ]; then
        echo -e "${YELLOW}仅检查: $only${NC}"
    fi

    # 执行检查
    if [ -z "$only" ] || [ "$only" = "system" ]; then check_system; fi
    if [ -z "$only" ] || [ "$only" = "gpu" ]; then check_gpu; fi
    if [ -z "$only" ] || [ "$only" = "conda" ]; then check_conda; fi
    if [ -z "$only" ] || [ "$only" = "hermes" ]; then check_hermes; fi
    if [ -z "$only" ] || [ "$only" = "vllm" ]; then check_vllm; fi
    if [ -z "$only" ] || [ "$only" = "leankit" ] || [ "$only" = "lean" ]; then check_leankit; fi
    if [ -z "$only" ] || [ "$only" = "network" ]; then check_network; fi

    # 汇总报告
    local total=$((PASS + FAIL))
    echo ""
    echo -e "${BLUE}══════════════════════════════════════${NC}"
    echo -e "  总计: $total 项 | ${GREEN}通过: $PASS${NC} | ${RED}失败: $FAIL${NC}"
    if [ $FAIL -eq 0 ]; then
        echo -e "  ${GREEN}✓ 所有检查通过！${NC}"
    else
        echo -e "  ${RED}✗ 存在 $FAIL 个失败项，日志保存在: $LOG_DIR${NC}"
    fi
    echo -e "${BLUE}══════════════════════════════════════${NC}"
    echo ""

    # 如果需要 Opus 诊断
    if $DIAGNOSE_MODE; then
        run_opus_diagnose
    elif [ $FAIL -gt 0 ]; then
        echo -e "${YELLOW}提示：运行 'check_env --diagnose' 让 Opus AI 自动诊断失败项${NC}"
        echo ""
    fi
}

main "$@"
