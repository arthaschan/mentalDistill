#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/../shared/common_env.sh"
resolve_python

RUN_ROOT="${RUN_ROOT:-$(ls -dt "$ROOT_DIR/runs/"* 2>/dev/null | head -1)}"
STUDENT_SIZE="${STUDENT_SIZE:-14b}"

if [[ -z "$RUN_ROOT" ]]; then
  echo "未找到 runs 目录。请通过 RUN_ROOT 指定。"
  exit 1
fi

echo "评估目录: $RUN_ROOT"
echo "学生模型: ${STUDENT_SIZE}B"

exec "$PY" "$ROOT_DIR/scripts/run_eval_dual.py" \
  --run_root "$RUN_ROOT" \
  --student_size "$STUDENT_SIZE"
