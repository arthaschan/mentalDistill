#!/usr/bin/env python3
"""fullEnglish 数据完整摘要报告 (纯 CPU, 写文件, 屏幕只打印一行)."""
import json
import os
from collections import Counter

DATA = "fullEnglish/00_data/out"
OUT = "fullEnglish/00_data/reports/data_summary.md"


def load(p):
    rows = []
    for line in open(p):
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


lines = ["# fullEnglish 数据摘要\n"]
for name in ["train", "val", "test_medqa", "test_medmcqa", "test_mmlu", "test_pubmedqa"]:
    rows = load(f"{DATA}/{name}.jsonl")
    src = Counter(r.get("source", "?") for r in rows)
    opt = Counter(r.get("n_options", "?") for r in rows)
    lines.append(f"\n## {name} (n={len(rows)})")
    lines.append(f"- 来源: {dict(src)}")
    lines.append(f"- 选项数: {dict(opt)}")
    if name == "test_mmlu":
        subj = Counter(r.get("subject", "?") for r in rows)
        lines.append(f"- 科目: {dict(sorted(subj.items()))}")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w").write("\n".join(lines) + "\n")
print(f"数据摘要已写入 {OUT}")
