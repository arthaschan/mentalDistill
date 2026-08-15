#!/usr/bin/env python3
"""fullEnglish 教师标签质量对比 (纯 CPU, 写文件).

对比 DeepSeek(API 硬标签) vs Llama70B(本地真实 logprobs) 在训练集上的标签质量:
  1. 各自教师准确率 (共同题上)
  2. 两教师一致率
  3. 分歧时的正确归属
  4. Llama70B 熵: 训练集 vs 测试集(筛选) 的难度对比
"""
import json
import os
from collections import Counter

L = "fullEnglish/03_main_distill/labels"
LP = "fullEnglish/01_teacher_screening/logprobs"
OUT = "fullEnglish/03_main_distill/reports/teacher_label_compare.md"


def load(path):
    rows = {}
    if not os.path.exists(path):
        return rows
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        uid = r.get("uid")
        gt = str(r.get("OriginalAnswer") or r.get("Answer", "")).strip().upper()
        pred = str(r.get("TeacherAnswer") or r.get("Answer", "")).strip().upper()
        if uid and gt in "ABCDE" and pred in "ABCDE":
            rows[uid] = {"gt": gt, "pred": pred,
                         "ent": r.get("TeacherEntropy"),
                         "src": r.get("source", "?")}
    return rows


ds = load(f"{L}/teacher_train.jsonl")
ll = load(f"{L}/teacher_train_llama70b.jsonl")
common = sorted(set(ds) & set(ll))

lines = ["# 教师标签质量对比 (训练集)\n"]
lines.append(f"- DeepSeek 已标注: {len(ds)} 题 | Llama70B 已标注: {len(ll)} 题 | 共同题: {len(common)}\n")

if common:
    ds_acc = 100 * sum(1 for u in common if ds[u]["pred"] == ds[u]["gt"]) / len(common)
    ll_acc = 100 * sum(1 for u in common if ll[u]["pred"] == ll[u]["gt"]) / len(common)
    agree = sum(1 for u in common if ds[u]["pred"] == ll[u]["pred"])
    lines.append("\n## 共同题上的教师准确率")
    lines.append(f"- DeepSeek: **{ds_acc:.2f}%**")
    lines.append(f"- Llama70B: **{ll_acc:.2f}%**")
    lines.append(f"- 两教师一致率: {100 * agree / len(common):.2f}% ({agree}/{len(common)})")
    # 分歧时的正确归属
    dis = [u for u in common if ds[u]["pred"] != ll[u]["pred"]]
    if dis:
        ds_win = sum(1 for u in dis if ds[u]["pred"] == ds[u]["gt"])
        ll_win = sum(1 for u in dis if ll[u]["pred"] == ll[u]["gt"])
        both_wrong = sum(1 for u in dis if ds[u]["pred"] != ds[u]["gt"] and ll[u]["pred"] != ll[u]["gt"])
        lines.append(f"\n## 分歧题 ({len(dis)} 题) 的正确归属")
        lines.append(f"- DeepSeek 对 / Llama70B 错: {ds_win}")
        lines.append(f"- Llama70B 对 / DeepSeek 错: {ll_win}")
        lines.append(f"- 都错: {both_wrong}")

# Llama70B 熵: 训练集 vs 测试集(筛选) 难度
train_ents = [ll[u]["ent"] for u in ll if ll[u]["ent"] is not None]
# 筛选 logprobs (600 题)
screen = load_all = {}
for f in [f"{LP}/Llama70B-AWQ_logprobs.jsonl"]:
    if os.path.exists(f):
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            uid = r.get("uid")
            ent = r.get("TeacherEntropy")
            if uid and ent is not None:
                screen[uid] = ent
if train_ents and screen:
    te = sum(train_ents) / len(train_ents)
    se = sum(screen.values()) / len(screen)
    lines.append("\n## Llama70B 平均熵 (难度代理)")
    lines.append(f"- 训练集: {te:.4f} (n={len(train_ents)})")
    lines.append(f"- 测试集(筛选): {se:.4f} (n={len(screen)})")
    lines.append(f"- 结论: {'测试集更难(熵更高)' if se > te else '训练集更难或相当'}")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"\n-> {OUT}")
