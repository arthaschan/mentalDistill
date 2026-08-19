#!/usr/bin/env python3
"""多教授并集 + 错误重叠分析（Phase 0 核心验证 P1/P2）。

输入：各教授的标签 JSONL（含 TeacherAnswer + OriginalAnswer + uid）。
输出：个体准确率、并集准确率(至少一个对)、最强单模型、并集增益、两两错误重叠。
支持部分覆盖（只用 uid 交集），教授数可变。
"""
import json
import sys
from itertools import combinations

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
    files = {
        "dsv4pro": "26_collaborative_distill/data/labels/dsv4pro_test_full.jsonl",
        "doubao": "26_collaborative_distill/data/labels/doubao_turbo_test_full.jsonl",
        "glm52": "26_collaborative_distill/data/labels/glm52_test_full.jsonl",
    }
    names = sys.argv[1:] if len(sys.argv) > 1 else ["dsv4pro", "doubao"]
    models = {n: load(files[n]) for n in names if n in files}

    common = set.intersection(*(set(m) for m in models.values()))
    print(f"教授数={len(models)} 交集题数={len(common)}", flush=True)

    if not common:
        print("[WARN] 无交集，先等标签生成", flush=True)
        return

    acc = {}
    for name, d in models.items():
        c = sum(1 for u in common if d[u][0] == d[u][1])
        acc[name] = 100.0 * c / len(common)

    # 并集：至少一个教授答对
    union_correct = sum(1 for u in common
                        if any(models[n][u][0] == models[n][u][1] for n in models))
    union = 100.0 * union_correct / len(common)
    best_name = max(acc, key=acc.get)
    best = acc[best_name]

    print("\n=== 个体准确率（交集口径） ===", flush=True)
    for n in names:
        print(f"  {n:10s}: {acc[n]:.2f}%", flush=True)

    print("\n=== 并集上限 ===", flush=True)
    print(f"  并集(至少一个对): {union:.2f}%", flush=True)
    print(f"  最强单模型({best_name}): {best:.2f}%", flush=True)
    print(f"  并集增益: +{union - best:.2f}pp", flush=True)

    print("\n=== 两两错误重叠（A 错时 B 也错的比例） ===", flush=True)
    for a, b in combinations(names, 2):
        a_wrong = [u for u in common if models[a][u][0] != models[a][u][1]]
        both = sum(1 for u in a_wrong if models[b][u][0] != models[b][u][1])
        comp = 100.0 * (len(a_wrong) - both) / len(a_wrong) if a_wrong else 0
        print(f"  {a} 错 {len(a_wrong)} 题 -> {b} 也错 {both} "
              f"({100.0*both/len(a_wrong):.1f}%, 互补 {comp:.1f}%)", flush=True)


if __name__ == "__main__":
    main()
