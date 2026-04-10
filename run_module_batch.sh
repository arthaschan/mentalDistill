#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODULES=(
  00_baseline_gt_sft
  01_qwen32b_whitebox
  02_deepseek_v3_choice_head
  03_doubao_choice_head
  04_kimi_choice_head
  05_qwen14b_choice_head
)

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
DEFAULT_LOG_DIR="${BATCH_LOG_DIR:-$ROOT_DIR/batch_logs/$TIMESTAMP}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"

usage() {
  cat <<'EOF'
Usage:
  bash rebuild/run_module_batch.sh ACTION [MODULE ...]

Actions:
  teacher-eval   Run scripts/run_teacher_eval.sh where available
  train          Run scripts/run_train.sh
  eval           Run scripts/run_eval.sh
  pipeline       Run scripts/run_pipeline.sh
  start-web      Start scripts/start_web.sh one by one (foreground)
  start-best-web Run scripts/start_best_web.sh one by one (foreground)
  list           Print available modules

Examples:
  bash rebuild/run_module_batch.sh list
  bash rebuild/run_module_batch.sh teacher-eval 01_qwen32b_whitebox 05_qwen14b_choice_head
  bash rebuild/run_module_batch.sh pipeline 00_baseline_gt_sft 02_deepseek_v3_choice_head
  CONTINUE_ON_ERROR=1 bash rebuild/run_module_batch.sh eval
EOF
}

ensure_module() {
  local module="$1"
  if [[ ! -d "$ROOT_DIR/$module" ]]; then
    echo "Unknown module: $module"
    exit 1
  fi
}

select_modules() {
  if [[ "$#" -gt 0 ]]; then
    SELECTED=("$@")
  else
    SELECTED=("${MODULES[@]}")
  fi

  for module in "${SELECTED[@]}"; do
    ensure_module "$module"
  done
}

run_action() {
  local action="$1"
  shift
  select_modules "$@"

  mkdir -p "$DEFAULT_LOG_DIR"
  local summary_file="$DEFAULT_LOG_DIR/summary.txt"
  : > "$summary_file"

  local script_name
  case "$action" in
    teacher-eval) script_name="run_teacher_eval.sh" ;;
    train) script_name="run_train.sh" ;;
    eval) script_name="run_eval.sh" ;;
    pipeline) script_name="run_pipeline.sh" ;;
    start-web) script_name="start_web.sh" ;;
    start-best-web) script_name="start_best_web.sh" ;;
    *)
      usage
      exit 1
      ;;
  esac

  local failures=0
  for module in "${SELECTED[@]}"; do
    local script_path="$ROOT_DIR/$module/scripts/$script_name"
    local log_path="$DEFAULT_LOG_DIR/${action}_${module}.log"
    echo ""
    echo "=== [$action] $module ==="
    if [[ ! -f "$script_path" ]]; then
      echo "skip: missing $script_name"
      printf 'SKIP\t%s\t%s\n' "$module" "$script_name" >> "$summary_file"
      continue
    fi
    if bash "$script_path" 2>&1 | tee "$log_path"; then
      printf 'OK\t%s\t%s\n' "$module" "$log_path" >> "$summary_file"
    else
      failures=$((failures + 1))
      printf 'FAIL\t%s\t%s\n' "$module" "$log_path" >> "$summary_file"
      if [[ "$CONTINUE_ON_ERROR" != "1" ]]; then
        echo "stopped on first failure; summary: $summary_file"
        exit 1
      fi
    fi
  done

  echo ""
  echo "Summary: $summary_file"
  cat "$summary_file"
  if [[ "$failures" -gt 0 ]]; then
    return 1
  fi
}

ACTION="${1:-}"
if [[ -z "$ACTION" ]]; then
  usage
  exit 1
fi
shift || true

case "$ACTION" in
  list)
    printf '%s\n' "${MODULES[@]}"
    ;;
  teacher-eval|train|eval|pipeline|start-web|start-best-web)
    run_action "$ACTION" "$@"
    ;;
  *)
    usage
    exit 1
    ;;
esac