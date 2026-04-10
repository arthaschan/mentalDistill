#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT_DIR/shared/common_env.sh"
resolve_python

echo "Python: $PY"
if [[ -n "${BASE_MODEL_7B:-}" || -d "${MODELS_DIR:-$ROOT_DIR/models}/Qwen2.5-7B-Instruct" ]]; then
  resolve_model_dir BASE_MODEL_7B Qwen2.5-7B-Instruct
  echo "7B model: $BASE_MODEL_7B"
fi
if [[ -n "${BASE_MODEL_14B:-}" || -d "${MODELS_DIR:-$ROOT_DIR/models}/Qwen2.5-14B-Instruct" ]]; then
  resolve_model_dir BASE_MODEL_14B Qwen2.5-14B-Instruct
  echo "14B model: $BASE_MODEL_14B"
fi
if [[ -n "${BASE_MODEL_32B:-}" || -d "${MODELS_DIR:-$ROOT_DIR/models}/Qwen2.5-32B-Instruct" ]]; then
  resolve_model_dir BASE_MODEL_32B Qwen2.5-32B-Instruct
  echo "32B model: $BASE_MODEL_32B"
fi

echo "Environment check OK"
