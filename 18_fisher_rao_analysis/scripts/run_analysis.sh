#!/usr/bin/env bash
set -euo pipefail
# Module 18: Fisher-Rao 信息几何分析 — 分析所有教师标签的几何特征
# 纯 CPU 计算，无需 GPU

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJ_ROOT="$(cd "$ROOT_DIR/.." && pwd)"
source "$PROJ_ROOT/shared/common_env.sh"
resolve_python

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="$ROOT_DIR/outputs/${TIMESTAMP}_fisher_rao"
mkdir -p "$OUTPUT_DIR"

echo "=== Module 18: Fisher-Rao Information Geometry Analysis ==="
echo "Output: $OUTPUT_DIR"
echo ""

# 收集所有可用的教师标签文件
TEACHERS=()

# Module 02: DeepSeek-V3 → 7B (83题数据集)
DEEPSEEK_7B="$PROJ_ROOT/02_deepseek_v3_choice_head/data/train_head_distill.jsonl"
if [[ -f "$DEEPSEEK_7B" ]]; then
    TEACHERS+=("DeepSeek-V3_7B:$DEEPSEEK_7B")
    echo "  ✓ DeepSeek-V3(7B data): $DEEPSEEK_7B"
fi

# Module 03: Doubao → 7B
DOUBAO_7B="$PROJ_ROOT/03_doubao_choice_head/data/train_head_distill.jsonl"
if [[ -f "$DOUBAO_7B" ]]; then
    TEACHERS+=("Doubao_7B:$DOUBAO_7B")
    echo "  ✓ Doubao(7B data): $DOUBAO_7B"
fi

# Module 04: Kimi → 7B (弱教师)
KIMI_7B="$PROJ_ROOT/04_kimi_choice_head/data/train_head_distill.jsonl"
if [[ -f "$KIMI_7B" ]]; then
    TEACHERS+=("Kimi_7B:$KIMI_7B")
    echo "  ✓ Kimi(7B data, weak teacher): $KIMI_7B"
fi

# Module 05: Qwen14B → 7B (同量级教师)
QWEN14B_7B="$PROJ_ROOT/05_qwen14b_choice_head/data/train_head_distill.jsonl"
if [[ -f "$QWEN14B_7B" ]]; then
    TEACHERS+=("Qwen14B_7B:$QWEN14B_7B")
    echo "  ✓ Qwen14B(7B data): $QWEN14B_7B"
fi

# Module 06: Qwen32B-API → 7B
QWEN32B_7B="$PROJ_ROOT/06_qwen32b_api_choice_head/data/train_head_distill.jsonl"
if [[ -f "$QWEN32B_7B" ]]; then
    TEACHERS+=("Qwen32B-API_7B:$QWEN32B_7B")
    echo "  ✓ Qwen32B-API(7B data): $QWEN32B_7B"
fi

echo ""
echo "Found ${#TEACHERS[@]} teacher label sets."
echo ""

if [[ ${#TEACHERS[@]} -lt 1 ]]; then
    echo "[ERROR] No teacher labels found. Please ensure teacher label files exist."
    exit 1
fi

# 运行分析
"$PY" "$PROJ_ROOT/shared/fisher_rao_analysis.py" \
    --teachers "${TEACHERS[@]}" \
    --output "$OUTPUT_DIR/fisher_rao_report.json" \
    2>&1 | tee "$ROOT_DIR/logs/fisher_rao_${TIMESTAMP}.log"

echo ""
echo "Report: $OUTPUT_DIR/fisher_rao_report.json"
echo "Log: $ROOT_DIR/logs/fisher_rao_${TIMESTAMP}.log"
