#!/usr/bin/env python3
"""恢复 MedMCQA 学科标签：按题目文本匹配原始 validation parquet 的 subject_name，写回 test_medmcqa.jsonl。"""
import json
from collections import Counter

import pandas as pd
from huggingface_hub import hf_hub_download

parquet = hf_hub_download(repo_id="openlifescienceai/medmcqa",
                          filename="data/validation-00000-of-00001.parquet",
                          repo_type="dataset")
df = pd.read_parquet(parquet)


def norm(s):
    return " ".join(str(s).split())


q2subj = {norm(q): s for q, s in zip(df["question"], df["subject_name"])}

path = "fullEnglish/00_data/out/test_medmcqa.jsonl"
rows = [json.loads(l) for l in open(path) if l.strip()]
m = 0
for r in rows:
    s = q2subj.get(norm(r.get("Question", "")))
    if s:
        r["subject"] = s
        m += 1

with open(path, "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"匹配写回: {m}/{len(rows)}")
c = Counter(r.get("subject", "(无)") for r in rows)
print("subject 分布:", c.most_common(15))
