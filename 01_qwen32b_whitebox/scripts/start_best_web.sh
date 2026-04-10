#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ADAPTER_DIR="${ADAPTER_DIR:-$ROOT_DIR/outputs/best}"

if [[ ! -d "$ADAPTER_DIR" ]]; then
  echo "未找到 whitebox 最佳 adapter: $ADAPTER_DIR"
  echo "请先训练，或通过 ADAPTER_DIR 指向具体 LoRA 目录。"
  exit 1
fi

echo "Using best adapter: $ADAPTER_DIR"
exec env ADAPTER_DIR="$ADAPTER_DIR" bash "$ROOT_DIR/scripts/start_web.sh"