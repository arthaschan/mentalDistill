#!/usr/bin/env bash
set -euo pipefail
# Module 17: α-divergence sweep — 在 DeepSeek 教师标签上测试不同 α 值
# α=1(KL前向), α=0.5, α=0(Hellinger), α=-1(KL反向), 各跑 2 seeds

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/../shared/common_env.sh"
resolve_python
resolve_model_dir BASE_MODEL_7B Qwen2.5-7B-Instruct

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_ROOT="$ROOT_DIR/runs/${TIMESTAMP}_alpha_sweep"
mkdir -p "$RUN_ROOT/logs"

# 复用 Module 02 的数据
TRAIN_HEAD="$ROOT_DIR/data/train_head_distill.jsonl"
VAL_DATA="$ROOT_DIR/data/val.jsonl"
TEST_DATA="$ROOT_DIR/data/test.jsonl"

PARAMS_FILE="$ROOT_DIR/configs/sweep_params.json"

echo "=== Module 17: α-divergence sweep ==="
echo "RUN_ROOT: $RUN_ROOT"
echo "TRAIN_HEAD: $TRAIN_HEAD"
echo ""

# 读取 JSON 配置，逐条执行
"$PY" -c "
import json, subprocess, sys, os

params = json.load(open('$PARAMS_FILE'))
run_root = '$RUN_ROOT'
py = '$PY'
base_model = '$BASE_MODEL_7B'
train_head = '$TRAIN_HEAD'
val_data = '$VAL_DATA'
test_data = '$TEST_DATA'
shared_dir = os.path.join('$ROOT_DIR', '..', 'shared')
script = os.path.join(shared_dir, 'train_alpha_distill.py')

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
        '--alpha', str(p['alpha_weight']),
        '--div_alpha', str(p['div_alpha']),
        '--default_distill_mask', '0',
        '--seed', str(p['seed']),
        '--deterministic',
    ]
    
    print(f'\\n[RUN] {name}: div_alpha={p[\"div_alpha\"]} ({p[\"div_alpha_label\"]}), seed={p[\"seed\"]}', flush=True)
    with open(log_file, 'w') as lf:
        rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT).returncode
    
    # 从日志中提取结果
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

print('\\n=== α-divergence sweep completed ===')
" 2>&1 | tee "$RUN_ROOT/logs/sweep_all.log"

echo ""
echo "Full logs: $RUN_ROOT/logs/"
echo "Results summary: grep TEST-BEST $RUN_ROOT/logs/train_*.log"
