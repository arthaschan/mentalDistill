#!/usr/bin/env python3
"""多教师加权投票 vs 并集 分析（Option B：2强 + 1弱 Llama）。

对比：
  - 个体准确率
  - 并集准确率（至少一个对，= 理论上限）
  - 简单多数投票（平局→最强教师）
  - 加权投票（权重 ∝ 准确率）
"""
import json
import sys
from collections import defaultdict

OPT = "ABCDE"


def load(path):
    d = {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        uid = r.get("uid")
        if not uid:
            continue
        ta = str(r.get("TeacherAnswer") or "").strip().upper()[:1]
        gt = str(r.get("OriginalAnswer") or r.get("Answer") or "").strip().upper()[:1]
        if ta in OPT and gt in OPT:
            d[uid] = (ta, gt)
    return d


def main():
    names = sys.argv[1:] if len(sys.argv) > 1 else \
        ["dsv4pro", "doubao", "llama70b"]
    files = {
        "dsv4pro": "26_collaborative_distill/data/labels/dsv4pro_test_full.jsonl",
        "doubao": "26_collaborative_distill/data/labels/doubao_turbo_test_full.jsonl",
        "llama70b": "26_collaborative_distill/data/labels/llama70b_test_full.jsonl",
    }
    models = {n: load(files[n]) for n in names if n in files and n in names}
    common = set.intersection(*(set(m) for m in models.values()))
    print(f"教授 {len(models)} 个, 交集 {len(common)} 题", flush=True)
    if not common:
        print("[WARN] 无交集", flush=True)
        return

    # 个体准确率
    acc = {}
    for n, d in models.items():
        acc[n] = 100.0 * sum(1 for u in common if d[u][0] == d[u][1]) / len(common)

    # 并集（至少一个对）
    union = 100.0 * sum(1 for u in common
                        if any(models[n][u][0] == models[n][u][1] for n in models)) \
        / len(common)

    # 加权投票（权重 ∝ 准确率）
    weights = {n: acc[n] for n in models}
    wvote_correct = 0
    for u in common:
        tally = defaultdict(float)
        for n in models:
            tally[models[n][u][0]] += weights[n]
        best_opt = max(tally, key=tally.get)
        if best_opt == models[list(models)[0]][u][1]:
            wvote_correct += 1
    wvote = 100.0 * wvote_correct / len(common)

    # 简单多数投票（平局→最强教师）
    strongest = max(acc, key=acc.get)
    mvote_correct = 0
    for u in common:
        tally = defaultdict(int)
        for n in models:
            tally[models[n][u][0]] += 1
        best_opt = max(tally, key=tally.get)
        # 平局时按最强教师
        if len([k for k in tally if tally[k] == tally[best_opt]]) > 1:
            best_opt = models[strongest][u][0]
        if best_opt == models[list(models)[0]][u][1]:
            mvote_correct += 1
    mvote = 100.0 * mvote_correct / len(common)

    print("\n=== 个体准确率 ===", flush=True)
    for n in names:
        print(f"  {n:10s}: {acc[n]:.2f}%", flush=True)
    print("\n=== 对比（交集口径） ===", flush=True)
    best = max(acc.values())
    print(f"  最强单模型: {best:.2f}%", flush=True)
    print(f"  并集(理论上限): {union:.2f}%  (+{union-best:.2f}pp)", flush=True)
    print(f"  简单多数投票: {mvote:.2f}%  (+{mvote-best:.2f}pp)", flush=True)
    print(f"  加权投票: {wvote:.2f}%  (+{wvote-best:.2f}pp)", flush=True)


if __name__ == "__main__":
    main()
