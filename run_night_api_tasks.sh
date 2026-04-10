#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$ROOT_DIR/.." && pwd)"
START_TIME="${NIGHT_START:-01:30}"
LOG_DIR="${NIGHT_LOG_DIR:-$ROOT_DIR/night_logs}"
mkdir -p "$LOG_DIR"

if [[ -f "$ROOT_DIR/setup.env" ]]; then
	# shellcheck disable=SC1091
	source "$ROOT_DIR/setup.env"
fi

CMD=$(cat <<EOF
cd "$PROJECT_ROOT"
export NIGHT_MODE=1
export REQUEST_INTERVAL_SEC="${REQUEST_INTERVAL_SEC:-8}"
export MAX_RETRIES="${MAX_RETRIES:-12}"
export RATE_LIMIT_COOLDOWN_SEC="${RATE_LIMIT_COOLDOWN_SEC:-300}"
export COOLDOWN_EVERY="${COOLDOWN_EVERY:-30}"
export COOLDOWN_SEC="${COOLDOWN_SEC:-120}"

bash "$ROOT_DIR/02_deepseek_v3_choice_head/scripts/generate_teacher_labels.sh"
bash "$ROOT_DIR/03_doubao_choice_head/scripts/generate_teacher_labels.sh"
export REQUEST_INTERVAL_SEC="${DOUBAO_SOFT_REQUEST_INTERVAL_SEC:-10}"
export RATE_LIMIT_COOLDOWN_SEC="${DOUBAO_SOFT_RATE_LIMIT_COOLDOWN_SEC:-420}"
export COOLDOWN_EVERY="${DOUBAO_SOFT_COOLDOWN_EVERY:-15}"
export COOLDOWN_SEC="${DOUBAO_SOFT_COOLDOWN_SEC:-180}"
bash "$ROOT_DIR/03_doubao_choice_head/scripts/generate_teacher_soft_labels.sh"
export REQUEST_INTERVAL_SEC="${REQUEST_INTERVAL_SEC:-8}"
export RATE_LIMIT_COOLDOWN_SEC="${RATE_LIMIT_COOLDOWN_SEC:-300}"
export COOLDOWN_EVERY="${COOLDOWN_EVERY:-30}"
export COOLDOWN_SEC="${COOLDOWN_SEC:-120}"
bash "$ROOT_DIR/04_kimi_choice_head/scripts/generate_teacher_labels.sh"
EOF
)

bash "$ROOT_DIR/shared/schedule_quiet_job.sh" "$START_TIME" "$LOG_DIR/night_api_tasks.log" "$CMD"
