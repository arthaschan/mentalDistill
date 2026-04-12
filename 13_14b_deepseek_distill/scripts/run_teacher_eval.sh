#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/../shared/common_env.sh"
resolve_python

SYSTEM_PROMPT_FILE="$SHARED_DIR/system_prompt_mcq.txt"
REQUEST_INTERVAL_SEC="${REQUEST_INTERVAL_SEC:-0}"
MAX_RETRIES="${MAX_RETRIES:-4}"
RATE_LIMIT_COOLDOWN_SEC="${RATE_LIMIT_COOLDOWN_SEC:-60}"
COOLDOWN_EVERY="${COOLDOWN_EVERY:-0}"
COOLDOWN_SEC="${COOLDOWN_SEC:-0}"

mkdir -p "$ROOT_DIR/logs"

"$PY" "$SHARED_DIR/generate_teacher_labels_api.py" \
  --candidate "$ROOT_DIR/configs/teacher_candidate.json" \
  --system_prompt "$SYSTEM_PROMPT_FILE" \
  --dataset "$ROOT_DIR/data/test.jsonl" \
  --output "$ROOT_DIR/data/teacher_test.jsonl" \
  --request_interval_sec "$REQUEST_INTERVAL_SEC" \
  --max_retries "$MAX_RETRIES" \
  --rate_limit_cooldown_sec "$RATE_LIMIT_COOLDOWN_SEC" \
  --cooldown_every "$COOLDOWN_EVERY" \
  --cooldown_sec "$COOLDOWN_SEC" \
  --resume \
  2>&1 | tee "$ROOT_DIR/logs/teacher_test.log"

"$PY" "$SHARED_DIR/teacher_eval_summary.py" \
  --input "$ROOT_DIR/data/teacher_test.jsonl" \
  --report "$ROOT_DIR/logs/teacher_eval_report.json" \
  2>&1 | tee "$ROOT_DIR/logs/teacher_eval_summary.log"
