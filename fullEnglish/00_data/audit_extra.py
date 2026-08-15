#!/usr/bin/env python3
"""fullEnglish 数据质量加深审计 (纯 CPU, 秒级).

检查:
  1. train 与各 test 的 uid 重叠 (数据泄漏)
  2. 答案字母分布 (是否偏)
  3. DeepSeek 教师标签质量 (教师-GT 一致率, 应约等于筛选的 81.67%)
"""
import json
import os
from collections import Counter

DATA = "fullEnglish/00_data/out"
LABELS = "fullEnglish/03_main_distill/labels"


def load(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


train = load(f"{DATA}/train.jsonl")
train_uids = {r.get("uid") for r in train}

tests = {n: load(f"{DATA}/{n}.jsonl") for n in
         ["test_medqa", "test_medmcqa", "test_mmlu", "test_pubmedqa"]}

print("=== 1. 数据泄漏检查 (train vs test uid 重叠) ===")
tot = 0
for name, rows in tests.items():
    test_uids = {r.get("uid") for r in rows}
    ov = train_uids & test_uids
    tot += len(ov)
    print(f"  {name:14s} 重叠 {len(ov)} / {len(rows)}")
print(f"  总重叠: {tot}  {'(0 = 无泄漏, 干净)' if tot == 0 else '(有泄漏! 需查)'}")

print("\n=== 2. 答案字母分布 ===")
for name, rows in [("train", train)] + [(n, r) for n, r in tests.items()]:
    c = Counter(r.get("Answer") for r in rows)
    n = len(rows)
    dist = "  ".join(f"{k}:{100 * v / n:.1f}%" for k, v in sorted(c.items()))
    print(f"  {name:14s} {dist}")

print("\n=== 3. DeepSeek 教师标签质量 ===")
lab = load(f"{LABELS}/teacher_train.jsonl")
if lab:
    n = len(lab)
    c = sum(1 for r in lab
            if str(r.get("OriginalAnswer", "")).strip().upper() == str(r.get("TeacherAnswer", "")).strip().upper())
    print(f"  已标注 {n} 题, 教师准确率 {100 * c / n:.2f}% (筛选零样本为 81.67%, 应接近)")
else:
    print("  (teacher_train.jsonl 尚未生成)")
