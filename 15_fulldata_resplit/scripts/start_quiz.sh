#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/../shared/common_env.sh"
resolve_python

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-7870}"
# 默认使用全量测试集(991题)，传 TEST_SET=dental 可切换为牙科子集(125题)
TEST_SET="${TEST_SET:-full}"
if [[ "$TEST_SET" == "dental" ]]; then
  TEST_FILE="$ROOT_DIR/data/test_dental.jsonl"
  TITLE="Module 15 牙科子集人机测试 (125题)"
else
  TEST_FILE="$ROOT_DIR/data/test.jsonl"
  TITLE="Module 15 全量人机测试 (991题)"
fi

exec "$PY" "$SHARED_DIR/quiz_app.py" \
  --test_data "$TEST_FILE" \
  --host "$HOST" \
  --port "$PORT" \
  --title "$TITLE"
