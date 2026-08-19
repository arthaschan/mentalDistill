#!/usr/bin/env python3
"""准备"无印度"实验的数据：
1. 训练集去掉 MedMCQA（印度）→ train_no_india.jsonl
2. 测试集去掉 MedMCQA → test_no_india.jsonl（medqa+mmlu）
3. 牙科子集（无印度）→ test_no_india_dental.jsonl
"""
import json
import re

DATA = "fullEnglish/00_data/out"
OUT = "26_collaborative_distill/data"

DENT = re.compile(
    r'\b(tooth|teeth|dental|dentine|dentin|enamel|pulp|molar|premolar|incisor|canine|'
    r'gingiv|periodont|oral|mandib|maxill|caries|occlus|denture|endodont|orthodont|'
    r'amalgam|prosthodont|alveolar|cementum|odonto|crown|root canal|fluoride|saliva|'
    r'palat|buccal|lingual|mucosa|periapical|dentition|bruxism|malocclusion)\b', re.I)


def load(path):
    rows = []
    for line in open(path):
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def is_dental(r):
    return bool(DENT.search(" ".join([str(r.get("Question", "")),
                                      str(r.get("Options", ""))])))


def main():
    # 1. 训练集去印度
    train = load(f"{DATA}/train.jsonl")
    train_ni = [r for r in train if r.get("source") != "medmcqa"]
    with open(f"{OUT}/train_no_india.jsonl", "w") as f:
        for r in train_ni:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"训练集: 原 {len(train)} → 无印度 {len(train_ni)}", flush=True)

    # 2. 测试集去印度（medqa + mmlu）
    test_ni = load(f"{DATA}/test_medqa.jsonl") + load(f"{DATA}/test_mmlu.jsonl")
    with open(f"{OUT}/test_no_india.jsonl", "w") as f:
        for r in test_ni:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"测试集(无印度): {len(test_ni)} 题", flush=True)

    # 3. 牙科子集（无印度）
    dental = [r for r in test_ni if is_dental(r)]
    with open(f"{OUT}/test_no_india_dental.jsonl", "w") as f:
        for r in dental:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"牙科子集(无印度): {len(dental)} 题", flush=True)


if __name__ == "__main__":
    main()
