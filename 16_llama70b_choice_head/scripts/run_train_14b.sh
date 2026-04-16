#!/usr/bin/env bash
set -euo pipefail
# Train 14B student with Llama-70B teacher (Stage 1 ONLY)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/../shared/common_env.sh"
resolve_python
resolve_model_dir BASE_MODEL_14B Qwen2.5-14B-Instruct
RUN_ROOT="$ROOT_DIR/runs/$(date +%Y%m%d_%H%M%S)_llama70b_14b"

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/outputs"

PARAMS_FILE="$ROOT_DIR/configs/grid_params_14b.json"
STAGE1_SCRIPT="$SHARED_DIR/train_choice_head_distill.py"

DATA_DIR="$ROOT_DIR/../15_fulldata_resplit/data"

"$PY" -c "
import json, subprocess, sys
from pathlib import Path

params = json.loads(Path('$PARAMS_FILE').read_text())
shared_dir = '$SHARED_DIR'
run_root = Path('$RUN_ROOT')

for p_cfg in params:
    name = f\"llama70b_14b_{p_cfg['name']}\"
    out_dir = run_root / 'outputs' / name / 'stage1_head'
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_root / 'logs' / f'stage1_{name}.log'

    cmd = [
        '$PY', '$STAGE1_SCRIPT',
        '--model_name', '$BASE_MODEL_14B',
        '--data_path', '$ROOT_DIR/data/train_head_distill.jsonl',
        '--val_path', '$DATA_DIR/val.jsonl',
        '--test_path', '$DATA_DIR/test.jsonl',
        '--output_dir', str(out_dir),
        '--num_epochs', str(p_cfg['num_epochs_stage1']),
        '--batch_size', str(p_cfg['batch_size']),
        '--gradient_accumulation_steps', str(p_cfg['gradient_accumulation_steps']),
        '--learning_rate', str(p_cfg['learning_rate_stage1']),
        '--rank', str(p_cfg['rank']),
        '--lora_alpha', str(p_cfg['lora_alpha']),
        '--alpha', str(p_cfg['alpha_stage1']),
        '--default_distill_mask', '0',
        '--seed', str(p_cfg['seed']),
        '--deterministic',
    ]

    print(f'[RUN] Stage1-only 14B: {name} seed={p_cfg[\"seed\"]}', flush=True)
    with log_path.open('w') as lf:
        rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        print(f'[FAIL] {name} rc={rc}', flush=True)
    else:
        print(f'[DONE] {name}', flush=True)
" 2>&1 | tee "$RUN_ROOT/logs/train_14b.log"
