#!/usr/bin/env bash
set -euo pipefail
# Generate DeepSeek teacher labels for the new full-data training set.
# Reuses existing labels from module 02 where available, only calls API for new questions.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/../shared/common_env.sh"
resolve_python
SYSTEM_PROMPT_FILE="$SHARED_DIR/system_prompt_mcq.txt"

REQUEST_INTERVAL_SEC="${REQUEST_INTERVAL_SEC:-1}"
MAX_RETRIES="${MAX_RETRIES:-6}"
RATE_LIMIT_COOLDOWN_SEC="${RATE_LIMIT_COOLDOWN_SEC:-120}"
COOLDOWN_EVERY="${COOLDOWN_EVERY:-50}"
COOLDOWN_SEC="${COOLDOWN_SEC:-30}"

mkdir -p "$ROOT_DIR/logs"

echo "=== Step 1: Generate labels for NEW questions (4077) ==="
"$PY" "$SHARED_DIR/generate_teacher_labels_api.py" \
  --candidate "$ROOT_DIR/configs/teacher_candidate.json" \
  --system_prompt "$SYSTEM_PROMPT_FILE" \
  --dataset "$ROOT_DIR/data/train_need_teacher.jsonl" \
  --output "$ROOT_DIR/data/teacher_train_new.jsonl" \
  --request_interval_sec "$REQUEST_INTERVAL_SEC" \
  --max_retries "$MAX_RETRIES" \
  --rate_limit_cooldown_sec "$RATE_LIMIT_COOLDOWN_SEC" \
  --cooldown_every "$COOLDOWN_EVERY" \
  --cooldown_sec "$COOLDOWN_SEC" \
  --resume \
  2>&1 | tee "$ROOT_DIR/logs/generate_teacher_new.log"

echo "=== Step 2: Merge old + new teacher labels ==="
"$PY" -c "
import json

# Load existing teacher labels from module 02
existing = {}
for f in ['$ROOT_DIR/../02_deepseek_v3_choice_head/data/teacher_train.jsonl',
          '$ROOT_DIR/../02_deepseek_v3_choice_head/data/teacher_test.jsonl']:
    try:
        with open(f) as fh:
            for line in fh:
                r = json.loads(line.strip())
                existing[r['Question']] = r
    except FileNotFoundError:
        pass
print(f'Existing teacher labels: {len(existing)}')

# Load new teacher labels
try:
    with open('$ROOT_DIR/data/teacher_train_new.jsonl') as fh:
        for line in fh:
            r = json.loads(line.strip())
            existing[r['Question']] = r
except FileNotFoundError:
    pass
print(f'Total teacher labels after merge: {len(existing)}')

# Load new train set and match
with open('$ROOT_DIR/data/train.jsonl') as fh:
    train = [json.loads(l.strip()) for l in fh if l.strip()]

matched = 0
with open('$ROOT_DIR/data/teacher_train.jsonl', 'w') as wf:
    for row in train:
        q = row['Question']
        if q in existing:
            wf.write(json.dumps(existing[q], ensure_ascii=False) + '\n')
            matched += 1
        # else: skip (no teacher label)
print(f'Matched {matched}/{len(train)} training questions with teacher labels')
" 2>&1 | tee -a "$ROOT_DIR/logs/generate_teacher_merge.log"

echo "=== Step 3: Generate labels for test set ==="
"$PY" "$SHARED_DIR/generate_teacher_labels_api.py" \
  --candidate "$ROOT_DIR/configs/teacher_candidate.json" \
  --system_prompt "$SYSTEM_PROMPT_FILE" \
  --dataset "$ROOT_DIR/data/test.jsonl" \
  --output "$ROOT_DIR/data/teacher_test.jsonl" \
  --request_interval_sec "$REQUEST_INTERVAL_SEC" \
  --max_retries "$MAX_RETRIES" \
  --rate_limit_cooldown_sec "$RATE_LIMIT_COOLDOWN_SEC" \
  --cooldown_every "$COOLDOWN_EVERY" \
  --cooldown_sec "$COOLDOWN_SEC" \
  --resume \
  2>&1 | tee "$ROOT_DIR/logs/generate_teacher_test.log"

echo "=== Done: teacher label generation ==="
