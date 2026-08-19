#!/usr/bin/env python3
"""任务A：扩大牙科测试集。从无印度训练集的牙科题里抽 320 道移入测试集。

产出：
- train_no_india_dentalsplit.jsonl（10488-320=10168，训练用，去掉抽出的牙科题）
- test_no_india_dental.jsonl（181+320=501，扩大后的牙科测试集）
"""
import json
import random
import re

DENT = re.compile(
    r'\b(tooth|teeth|dental|dentine|dentin|enamel|pulp|molar|premolar|incisor|canine|'
    r'gingiv|periodont|oral|mandib|maxill|caries|occlus|denture|endodont|orthodont|'
    r'amalgam|prosthodont|alveolar|cementum|odonto|crown|root canal|fluoride|saliva|'
    r'palat|buccal|lingual|mucosa|periapical|dentition|bruxism|malocclusion)\b', re.I)

PULL = 320
SEED = 42


def is_dental(r):
    return bool(DENT.search(" ".join([str(r.get("Question", "")),
                                      str(r.get("Options", ""))])))


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def main():
    train = load("26_collaborative_distill/data/train_no_india.jsonl")
    dental = [r for r in train if is_dental(r)]
    nondental = [r for r in train if not is_dental(r)]
    print(f"训练集 {len(train)} 题, 牙科 {len(dental)}, 非牙科 {len(nondental)}", flush=True)

    random.seed(SEED)
    pulled = random.sample(dental, PULL)
    pulled_uids = {r["uid"] for r in pulled}
    kept = [r for r in dental if r["uid"] not in pulled_uids]
    print(f"抽出 {len(pulled)} 道牙科进测试, 训练保留牙科 {len(kept)}", flush=True)

    new_train = nondental + kept
    with open("26_collaborative_distill/data/train_no_india_dentalsplit.jsonl", "w") as f:
        for r in new_train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"新训练集: {len(new_train)} 题", flush=True)

    # 扩大牙科测试集：原 181 + 抽出 320
    old_dental_test = load("26_collaborative_distill/data/test_no_india_dental.jsonl")
    new_dental_test = old_dental_test + pulled
    with open("26_collaborative_distill/data/test_no_india_dental.jsonl", "w") as f:
        for r in new_dental_test:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"新牙科测试集: {len(old_dental_test)} + {len(pulled)} = {len(new_dental_test)} 题",
          flush=True)


if __name__ == "__main__":
    main()
