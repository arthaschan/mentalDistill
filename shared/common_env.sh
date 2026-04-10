#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${ROOT_DIR:-}" ]]; then
  echo "ROOT_DIR is required before sourcing common_env.sh"
  exit 1
fi

REBUILD_ROOT="$(cd "$ROOT_DIR/.." && pwd)"
SHARED_DIR="$REBUILD_ROOT/shared"
MODELS_DIR="${MODELS_DIR:-$REBUILD_ROOT/models}"

resolve_python() {
  if [[ -n "${EASYEDIT_PY:-}" ]]; then
    PY="$EASYEDIT_PY"
  elif command -v python3 >/dev/null 2>&1; then
    PY="$(command -v python3)"
  else
    echo "python3 not found; set EASYEDIT_PY"
    exit 1
  fi
}

resolve_model_dir() {
  local target_var="$1"
  local model_dir_name="$2"
  local configured_value="${!target_var:-}"
  if [[ -n "$configured_value" ]]; then
    printf -v "$target_var" '%s' "$configured_value"
    return 0
  fi
  if [[ -d "$MODELS_DIR/$model_dir_name" ]]; then
    printf -v "$target_var" '%s' "$MODELS_DIR/$model_dir_name"
    return 0
  fi
  echo "Missing model directory for $target_var. Expected: $MODELS_DIR/$model_dir_name or env $target_var"
  exit 1
}
