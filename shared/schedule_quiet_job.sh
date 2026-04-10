#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "用法: $0 HH:MM LOG_FILE COMMAND..."
  echo "示例: $0 01:30 rebuild/night_logs/deepseek.log bash rebuild/02_deepseek_v3_choice_head/scripts/generate_teacher_labels.sh"
  exit 1
fi

START_TIME="$1"
shift
LOG_FILE="$1"
shift

mkdir -p "$(dirname "$LOG_FILE")"

DELAY_SEC="$(python3 - "$START_TIME" <<'PY'
import sys
from datetime import datetime, timedelta
hh, mm = map(int, sys.argv[1].split(':'))
now = datetime.now()
target = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
if target <= now:
    target += timedelta(days=1)
print(int((target - now).total_seconds()))
PY
)"

nohup bash -lc "sleep $DELAY_SEC; $*" >> "$LOG_FILE" 2>&1 &
PID=$!
echo "scheduled pid=$PID start_at=$START_TIME log=$LOG_FILE"
