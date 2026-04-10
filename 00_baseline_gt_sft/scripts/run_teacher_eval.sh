#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT_DIR/logs"

printf 'Baseline project has no teacher model.\n' | tee "$ROOT_DIR/logs/teacher_eval.log"