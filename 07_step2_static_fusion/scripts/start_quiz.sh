#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/../shared/common_env.sh"
resolve_python

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-7870}"
# 融合/集成模块使用 Module 00 的测试集 (83题牙科)
TEST_FILE="${TEST_FILE:-$ROOT_DIR/../00_baseline_gt_sft/data/test.jsonl}"

exec "$PY" "$SHARED_DIR/quiz_app.py" \
  --test_data "$TEST_FILE" \
  --host "$HOST" \
  --port "$PORT" \
  --title "$(basename "$ROOT_DIR") 人机测试"
