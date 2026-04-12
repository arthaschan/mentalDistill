#!/usr/bin/env bash
# =============================================================================
# 阶段0 数据准备脚本
# - 等待先前定时任务完成 (PID 666389)
# - 复制 EasyEdit3 现有 teacher 标签到 mentalDistill
# - 使用 Qwen-32B 本地生成 teacher 标签 (含软标签)
# - 构建 RAG 索引 (仅主库 CMExam Explanation)
# - 忽略次库 (教材 PDF) 和 Huatuo 补充
# =============================================================================
set -euo pipefail

PROJ_DIR="/home/student/arthas/mentalDistill"
EE3_DIR="/home/student/arthas/EasyEdit3"
PYTHON="/home/student/anaconda3/bin/python3"
LOG_DIR="$PROJ_DIR/night_logs"
LOG_FILE="$LOG_DIR/stage0_prep.log"

mkdir -p "$LOG_DIR"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================"
echo "阶段0 数据准备 - $(date)"
echo "========================================"

# ---- Step A: 等待 PID 666389 (API task) 完成 ----
API_PID=666389
if kill -0 $API_PID 2>/dev/null; then
    echo "[WAIT] PID $API_PID still running, waiting..."
    while kill -0 $API_PID 2>/dev/null; do
        sleep 30
    done
    echo "[WAIT] PID $API_PID finished at $(date)"
else
    echo "[WAIT] PID $API_PID already finished"
fi

echo ""
echo "========================================"
echo "Step A: 先前任务已完成，开始阶段0"
echo "========================================"

# ---- Step B: 复制 teacher 标签 ----
echo ""
echo "========================================"
echo "Step B: 复制 teacher 标签从 EasyEdit3"
echo "========================================"

# DeepSeek-V3: 硬标签 672 条
SRC="$EE3_DIR/rebuild/02_deepseek_v3_choice_head/data/teacher_train.jsonl"
DST="$PROJ_DIR/02_deepseek_v3_choice_head/data/teacher_train.jsonl"
if [[ -s "$SRC" ]]; then
    cp "$SRC" "$DST"
    echo "[OK] DeepSeek-V3 hard labels: $(wc -l < "$DST") lines → $DST"
else
    echo "[WARN] DeepSeek-V3 hard labels not found at $SRC"
fi

# Doubao: 硬标签 672 条 (from distill_runs)
SRC="$EE3_DIR/external_model_benchmark_20260326/distill_runs/doubao/artifacts/teacher_train.jsonl"
DST="$PROJ_DIR/03_doubao_choice_head/data/teacher_train.jsonl"
if [[ -s "$SRC" ]]; then
    cp "$SRC" "$DST"
    echo "[OK] Doubao hard labels: $(wc -l < "$DST") lines → $DST"
else
    echo "[WARN] Doubao hard labels not found at $SRC"
fi

# Doubao: 软标签 604 条
SRC="$EE3_DIR/rebuild/03_doubao_choice_head/data/teacher_train_soft.jsonl"
DST="$PROJ_DIR/03_doubao_choice_head/data/teacher_train_soft.jsonl"
if [[ -s "$SRC" ]]; then
    cp "$SRC" "$DST"
    echo "[OK] Doubao soft labels: $(wc -l < "$DST") lines → $DST"
else
    echo "[WARN] Doubao soft labels not found at $SRC"
fi

# Kimi: 硬标签 672 条 (纳入对照，主线不使用)
SRC="$EE3_DIR/rebuild/04_kimi_choice_head/data/teacher_train.jsonl"
DST="$PROJ_DIR/04_kimi_choice_head/data/teacher_train.jsonl"
if [[ -s "$SRC" ]]; then
    cp "$SRC" "$DST"
    echo "[OK] Kimi hard labels: $(wc -l < "$DST") lines → $DST"
else
    echo "[WARN] Kimi hard labels not found at $SRC"
fi

# Qwen-14B: 硬标签 + 软标签 672 条
SRC="$EE3_DIR/rebuild/05_qwen14b_choice_head/data/teacher_train.jsonl"
DST="$PROJ_DIR/05_qwen14b_choice_head/data/teacher_train.jsonl"
if [[ -s "$SRC" ]]; then
    cp "$SRC" "$DST"
    echo "[OK] Qwen-14B hard labels: $(wc -l < "$DST") lines → $DST"
else
    echo "[WARN] Qwen-14B hard labels not found at $SRC"
fi

SRC="$EE3_DIR/rebuild/05_qwen14b_choice_head/data/teacher_train_soft.jsonl"
DST="$PROJ_DIR/05_qwen14b_choice_head/data/teacher_train_soft.jsonl"
if [[ -s "$SRC" ]]; then
    cp "$SRC" "$DST"
    echo "[OK] Qwen-14B soft labels: $(wc -l < "$DST") lines → $DST"
else
    echo "[WARN] Qwen-14B soft labels not found at $SRC"
fi

# Qwen-14B: teacher_test
SRC="$EE3_DIR/rebuild/05_qwen14b_choice_head/data/teacher_test.jsonl"
DST="$PROJ_DIR/05_qwen14b_choice_head/data/teacher_test.jsonl"
if [[ -s "$SRC" ]]; then
    cp "$SRC" "$DST"
    echo "[OK] Qwen-14B teacher_test: $(wc -l < "$DST") lines → $DST"
fi

echo ""
echo "========================================"
echo "Step C: 为缺失软标签的老师生成 TeacherDist"
echo "========================================"

# DeepSeek-V3: hard → soft (label smoothing)
echo "[SOFT] DeepSeek-V3: hard → soft label smoothing..."
$PYTHON "$PROJ_DIR/shared/prepare_soft_labels.py" \
    --input "$PROJ_DIR/02_deepseek_v3_choice_head/data/teacher_train.jsonl" \
    --output "$PROJ_DIR/02_deepseek_v3_choice_head/data/teacher_train_soft.jsonl" \
    --smooth_eps 0.25 || echo "[WARN] DeepSeek soft label conversion failed"

# Doubao: hard → soft (补全 604→672)
echo "[SOFT] Doubao: hard → soft label smoothing (补全至672条)..."
$PYTHON "$PROJ_DIR/shared/prepare_soft_labels.py" \
    --input "$PROJ_DIR/03_doubao_choice_head/data/teacher_train.jsonl" \
    --output "$PROJ_DIR/03_doubao_choice_head/data/teacher_train_soft_full.jsonl" \
    --smooth_eps 0.25 || echo "[WARN] Doubao soft label completion failed"

# Kimi: hard → soft (备用对照)
echo "[SOFT] Kimi: hard → soft label smoothing (对照用)..."
$PYTHON "$PROJ_DIR/shared/prepare_soft_labels.py" \
    --input "$PROJ_DIR/04_kimi_choice_head/data/teacher_train.jsonl" \
    --output "$PROJ_DIR/04_kimi_choice_head/data/teacher_train_soft.jsonl" \
    --smooth_eps 0.25 || echo "[WARN] Kimi soft label conversion failed"

echo ""
echo "========================================"
echo "Step D: 本地 Qwen-32B 生成 teacher 标签 (含软标签)"
echo "========================================"

MODEL_32B="/home/student/arthas/EasyEdit3/Qwen2.5-32B-Instruct"
TRAIN_DATA="$PROJ_DIR/01_qwen32b_whitebox/data/train.jsonl"
TEACHER_OUT="$PROJ_DIR/01_qwen32b_whitebox/data/teacher_train.jsonl"
TEACHER_SOFT_OUT="$PROJ_DIR/01_qwen32b_whitebox/data/teacher_train_soft.jsonl"

if [[ -d "$MODEL_32B" ]]; then
    echo "[32B] Generating teacher labels with Qwen2.5-32B-Instruct..."
    echo "[32B] Model: $MODEL_32B"
    echo "[32B] Data: $TRAIN_DATA ($(wc -l < "$TRAIN_DATA") lines)"

    # 生成硬标签
    $PYTHON "$PROJ_DIR/shared/generate_teacher_labels_local.py" \
        --model_path "$MODEL_32B" \
        --dataset "$TRAIN_DATA" \
        --output "$TEACHER_OUT" || {
        echo "[ERR] Qwen-32B teacher label generation failed"
    }

    # 从硬标签生成软标签
    if [[ -s "$TEACHER_OUT" ]]; then
        echo "[32B] Hard labels: $(wc -l < "$TEACHER_OUT") lines"
        $PYTHON "$PROJ_DIR/shared/prepare_soft_labels.py" \
            --input "$TEACHER_OUT" \
            --output "$TEACHER_SOFT_OUT" \
            --smooth_eps 0.25 || echo "[WARN] 32B soft label conversion failed"
    fi
else
    echo "[WARN] Qwen-32B model not found at $MODEL_32B, skipping"
fi

echo ""
echo "========================================"
echo "Step E: 构建 RAG 主库索引 (CMExam Explanation)"
echo "========================================"

$PYTHON - <<'PYEOF'
import json
import os
import pickle
import numpy as np
from pathlib import Path

PROJ_DIR = "/home/student/arthas/mentalDistill"
RAG_DIR = os.path.join(PROJ_DIR, "shared", "rag_index")
os.makedirs(RAG_DIR, exist_ok=True)

# 收集所有 train/val/test 中唯一的 (Question, Explanation) 对
seen_q = set()
docs = []

for module in ["00_baseline_gt_sft", "01_qwen32b_whitebox", "02_deepseek_v3_choice_head",
               "03_doubao_choice_head", "04_kimi_choice_head", "05_qwen14b_choice_head"]:
    for split in ["train.jsonl", "val.jsonl", "test.jsonl"]:
        fp = os.path.join(PROJ_DIR, module, "data", split)
        if not os.path.exists(fp):
            continue
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                q = row.get("Question", "").strip()
                if q in seen_q:
                    continue
                seen_q.add(q)
                exp = row.get("Explanation", "").strip()
                if not exp:
                    continue
                meta = {
                    "disease_group": row.get("Disease Group", ""),
                    "department": row.get("Clinical Department", ""),
                    "discipline": row.get("Medical Discipline", ""),
                    "difficulty": row.get("Difficulty level", ""),
                }
                docs.append({
                    "question": q,
                    "explanation": exp,
                    "metadata": meta,
                })

print(f"[RAG] Collected {len(docs)} unique (Question, Explanation) pairs")

# 用 sentence-transformers 生成 embedding
try:
    from sentence_transformers import SentenceTransformer
    model_name = "all-MiniLM-L6-v2"
    print(f"[RAG] Loading embedding model: {model_name}")
    embed_model = SentenceTransformer(model_name)

    texts = [d["explanation"] for d in docs]
    print(f"[RAG] Encoding {len(texts)} texts...")
    embeddings = embed_model.encode(texts, show_progress_bar=True, batch_size=64)
    embeddings = np.array(embeddings, dtype=np.float32)

    # 保存 index
    np.save(os.path.join(RAG_DIR, "embeddings.npy"), embeddings)
    with open(os.path.join(RAG_DIR, "docs.json"), "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

    print(f"[RAG] Saved embeddings: {embeddings.shape} → {RAG_DIR}/embeddings.npy")
    print(f"[RAG] Saved docs: {len(docs)} → {RAG_DIR}/docs.json")
    print("[RAG] RAG 主库索引构建完成 (CMExam Explanation only)")
    print("[RAG] 次库(教材PDF)和补充(Huatuo)已按要求跳过")

except ImportError as e:
    print(f"[WARN] sentence-transformers not available: {e}")
    # 退回到简单的 JSON 导出
    with open(os.path.join(RAG_DIR, "docs.json"), "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)
    print(f"[RAG] Saved docs only (no embeddings): {len(docs)} → {RAG_DIR}/docs.json")
    print("[RAG] 安装 sentence-transformers 后可重新生成 embedding")

PYEOF

echo ""
echo "========================================"
echo "Step F: 数据完备性检查"
echo "========================================"

echo ""
echo "Teacher 标签清单:"
echo "--------------------------------------------"
for module in 01_qwen32b_whitebox 02_deepseek_v3_choice_head 03_doubao_choice_head 04_kimi_choice_head 05_qwen14b_choice_head; do
    echo "[$module]"
    for f in teacher_train.jsonl teacher_train_soft.jsonl teacher_train_soft_full.jsonl teacher_test.jsonl; do
        fp="$PROJ_DIR/$module/data/$f"
        if [[ -f "$fp" ]]; then
            lines=$(wc -l < "$fp")
            echo "  $f: $lines lines"
        fi
    done
done

echo ""
echo "RAG 索引:"
echo "--------------------------------------------"
ls -la "$PROJ_DIR/shared/rag_index/" 2>/dev/null || echo "  (not built)"

echo ""
echo "========================================"
echo "阶段0 数据准备完成 - $(date)"
echo "========================================"
echo ""
echo "=== 缺失项 (需人工处理) ==="
echo "1. setup.env 中 API keys 为空 → API 老师标签生成不可用"
echo "   DEEPSEEK_API_KEY, DOUBAO_API_KEY, MOONSHOT_API_KEY"
echo "2. Doubao 软标签原始仅 604/672 条 (已用 hard→soft 补全至 672)"
echo "3. Qwen-32B 软标签来自 label smoothing (非真实 logits)"
echo ""
echo "=== 下一步: Step 1 多学生投票集成 ==="
echo "  需要各模块已训练好的 LoRA 适配器"
echo "  检查 EasyEdit3 中是否有现成 LoRA..."
find /home/student/arthas/EasyEdit3/rebuild -maxdepth 4 -name "adapter_config.json" -path "*/best/*" 2>/dev/null
