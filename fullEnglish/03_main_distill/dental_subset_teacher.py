#!/usr/bin/env python3
"""识别全科测试集中的牙科题, 统计老师(DeepSeek)在牙科子集的准确率 (纯 CPU).

牙科识别 = 关键词匹配 (复用英文牙科实验 extract_medmcqa_dental.py 的 DENT 正则,
因 fullEnglish 转换时丢弃了 MedMCQA 的 subject_name).
"""
import json
import os
import re
from collections import Counter

DENT = re.compile(
    r'\b(tooth|teeth|dental|dentine|dentin|enamel|pulp|molar|premolar|incisor|canine|'
    r'gingiv|periodont|oral|mandib|maxill|caries|occlus|denture|endodont|orthodont|'
    r'amalgam|prosthodont|alveolar|cementum|odonto|crown|root canal|fluoride|saliva|'
    r'palat|buccal|lingual|mucosa|periapical|dentition|bruxism|malocclusion)\b', re.I)

DATA = "fullEnglish/00_data/out"
LABELS = "fullEnglish/03_main_distill/labels"


def load(path):
    rows = []
    for line in open(path):
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def is_dental(r):
    txt = " ".join([str(r.get("Question", "")), str(r.get("Options", ""))])
    return bool(DENT.search(txt))


# 老师标签 (DeepSeek) 按 uid 索引
teacher = {}
for s in ["test_medqa", "test_medmcqa", "test_mmlu"]:
    p = f"{LABELS}/teacher_{s}.jsonl"
    if os.path.exists(p):
        for r in load(p):
            uid = r.get("uid")
            if uid:
                teacher[uid] = r

report = {}
total_dental = 0
for s in ["test_medqa", "test_medmcqa", "test_mmlu"]:
    rows = load(f"{DATA}/{s}.jsonl")
    dental = [r for r in rows if is_dental(r)]
    n = len(dental)
    total_dental += n
    # 老师准确率 (牙科子集)
    c = 0
    for r in dental:
        tr = teacher.get(r.get("uid"))
        if tr:
            gt = str(tr.get("OriginalAnswer") or r.get("Answer", "")).strip().upper()
            ta = str(tr.get("TeacherAnswer") or tr.get("Answer", "")).strip().upper()
            if gt in "ABCDE" and ta in "ABCDE" and ta == gt:
                c += 1
    t_acc = round(100 * c / n, 2) if n else None
    report[s] = {"n_dental": n, "teacher_acc": t_acc, "total": len(rows)}
    print(f"{s:14s} 总 {len(rows):5d}  牙科 {n:4d}  教师准确率 {t_acc}%")

print(f"\n牙科题合计: {total_dental}")
json.dump(report, open("fullEnglish/03_main_distill/reports/dental_subset_teacher.json", "w"),
          ensure_ascii=False, indent=2)
print("-> fullEnglish/03_main_distill/reports/dental_subset_teacher.json")
