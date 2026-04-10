#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

bash "$ROOT_DIR/scripts/generate_teacher_labels.sh"
bash "$ROOT_DIR/scripts/prepare_soft_labels.sh"
bash "$ROOT_DIR/scripts/build_head_dataset.sh"
bash "$ROOT_DIR/scripts/run_train.sh"
bash "$ROOT_DIR/scripts/run_eval.sh"