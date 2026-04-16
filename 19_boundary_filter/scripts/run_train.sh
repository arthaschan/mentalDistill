#!/usr/bin/env bash
set -euo pipefail
# Module 19: Fisher 边界距离置信度过滤蒸馏 sweep

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/../shared/common_env.sh"
resolve_python
resolve_model_dir BASE_MODEL_7B Qwen2.5-7B-Instruct

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_ROOT="$ROOT_DIR/runs/${TIMESTAMP}_boundary_sweep"
mkdir -p "$RUN_ROOT/logs"

# 使用 Module 16 Llama-70B 真实 logprobs 数据（边界距离分布丰富）
# smooth_eps 数据的边界距离全部相同(0.4510)，过滤无意义
TRAIN_HEAD="$ROOT_DIR/../16_llama70b_choice_head/data/train_head_distill.jsonl"
VAL_DATA="$ROOT_DIR/../15_fulldata_resplit/data/val.jsonl"
TEST_DATA="$ROOT_DIR/../15_fulldata_resplit/data/test.jsonl"

PARAMS_FILE="$ROOT_DIR/configs/sweep_params.json"
SCRIPT="$SHARED_DIR/train_boundary_filter_distill.py"

echo "=== Module 19: Boundary-Distance Confidence Filtering Sweep ==="
echo "RUN_ROOT: $RUN_ROOT"
echo ""

"$PY" -c "
import json, subprocess, sys, os

params = json.load(open('$PARAMS_FILE'))
run_root = '$RUN_ROOT'
py = '$PY'
base_model = '$BASE_MODEL_7B'
train_head = '$TRAIN_HEAD'
val_data = '$VAL_DATA'
test_data = '$TEST_DATA'
script = '$SCRIPT'

for p in params:
    name = p['name']
    out_dir = os.path.join(run_root, 'outputs', name)
    log_file = os.path.join(run_root, 'logs', f'train_{name}.log')

    cmd = [
        py, script,
        '--model_name', base_model,
        '--data_path', train_head,
        '--val_path', val_data,
        '--test_path', test_data,
        '--output_dir', out_dir,
        '--num_epochs', str(p['num_epochs']),
        '--batch_size', str(p['batch_size']),
        '--gradient_accumulation_steps', str(p['gradient_accumulation_steps']),
        '--learning_rate', str(p['learning_rate']),
        '--rank', str(p['rank']),
        '--lora_alpha', str(p['lora_alpha']),
        '--alpha', str(p['alpha']),
        '--boundary_threshold', str(p['boundary_threshold']),
        '--filter_mode', p['filter_mode'],
        '--seed', str(p['seed']),
        '--deterministic',
    ]
    if p['filter_mode'] == 'soft' and 'soft_gamma' in p:
        cmd.extend(['--soft_gamma', str(p['soft_gamma'])])

    print(f'\\n[RUN] {name}: threshold={p[\"boundary_threshold\"]}, mode={p[\"filter_mode\"]}, seed={p[\"seed\"]}', flush=True)
    with open(log_file, 'w') as lf:
        rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT).returncode

    best_val = best_test = 'N/A'
    with open(log_file, 'r') as lf:
        for line in lf:
            if '[VAL]' in line:
                best_val = line.strip()
            if '[TEST-BEST]' in line:
                best_test = line.strip()

    status = 'OK' if rc == 0 else f'FAIL(rc={rc})'
    print(f'  [{status}] {best_val}')
    print(f'  [{status}] {best_test}')

print('\\n=== Boundary filter sweep completed ===')
" 2>&1 | tee "$RUN_ROOT/logs/sweep_all.log"

echo ""
echo "Full logs: $RUN_ROOT/logs/"
echo "Results: grep TEST-BEST $RUN_ROOT/logs/train_*.log"
