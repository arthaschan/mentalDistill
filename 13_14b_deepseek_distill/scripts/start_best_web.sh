#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "${ADAPTER_DIR:-}" ]]; then
  LATEST_RUN="$(ls -td "$ROOT_DIR"/runs/* 2>/dev/null | head -1)"
  if [[ -z "$LATEST_RUN" ]]; then
    echo "未找到 runs 目录，请先训练或通过 ADAPTER_DIR 指定。"
    exit 1
  fi
  ADAPTER_DIR="$(find "$LATEST_RUN/outputs" -type d -path '*/stage2_sft/best' | head -1)"
fi

if [[ -z "$ADAPTER_DIR" || ! -d "$ADAPTER_DIR" ]]; then
  echo "未找到 14B DeepSeek 最佳 stage2 adapter，请检查训练结果。"
  exit 1
fi

echo "Using best adapter: $ADAPTER_DIR"
exec env ADAPTER_DIR="$ADAPTER_DIR" bash "$ROOT_DIR/scripts/start_web.sh"
