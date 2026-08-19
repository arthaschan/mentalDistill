#!/usr/bin/env python3
"""gating/置信度路由实验：能否用 Llama 的置信度(熵)做智能路由，抓住并集的 +2pp。

对比基线：强教师投票 88.60%、并集(理论上限) 93.68%。
路由规则：默认信强教师，仅当 Llama 高置信(低熵)且与强教师答案不同时，改信 Llama。
扫 Llama 熵阈值 θ 找最优。
"""
import json
import sys
from collections import defaultdict

OPT = "ABCDE"


def load(path, with_dist=False):
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
        if ta not in OPT or gt not in OPT:
            continue
        if with_dist:
            ent = float(r.get("TeacherEntropy", 0.0))
            d[uid] = (ta, gt, ent)
        else:
            d[uid] = (ta, gt)
    return d


def main():
    dsv = load("26_collaborative_distill/data/labels/dsv4pro_test_full.jsonl")
    dob = load("26_collaborative_distill/data/labels/doubao_turbo_test_full.jsonl")
    llama = load("26_collaborative_distill/data/labels/llama70b_test_full.jsonl",
                 with_dist=True)
    common = set(dsv) & set(dob) & set(llama)
    print(f"交集 {len(common)} 题", flush=True)

    # 基线
    strong_acc = 100.0 * sum(1 for u in common if dsv[u][0] == dsv[u][1]) / len(common)
    union = 100.0 * sum(1 for u in common if dsv[u][0] == dsv[u][1]
                        or dob[u][0] == dob[u][1] or llama[u][0] == llama[u][1]) / len(common)
    print(f"基线: 最强单模型 {strong_acc:.2f}%  并集(上限) {union:.2f}%", flush=True)

    # 路由规则：默认 dsv4pro（最强），当 Llama 熵 < θ 且 Llama 答 != dsv 答时改信 Llama
    print("\n=== 规则A: 熵阈值路由（默认dsv4pro，Llama高置信且分歧时改信Llama） ===", flush=True)
    for th in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.0, 1.2]:
        c = 0
        for u in common:
            ans = dsv[u][0]
            if llama[u][2] < th and llama[u][0] != dsv[u][0]:
                ans = llama[u][0]
            if ans == dsv[u][1]:
                c += 1
        print(f"  θ<{th:4.2f}: {100.0*c/len(common):.2f}%", flush=True)

    # 规则B：默认"强教师一致则用、分歧则 Llama"，再叠加 Llama 高置信覆盖
    print("\n=== 规则B: 强教师一致→用；分歧→Llama；再叠加 Llama高置信覆盖一致答案 ===", flush=True)
    for th in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]:
        c = 0
        for u in common:
            if dsv[u][0] == dob[u][0]:
                ans = dsv[u][0]
                # 强教师一致但 Llama 高置信且不同 → 覆盖
                if llama[u][2] < th and llama[u][0] != ans:
                    ans = llama[u][0]
            else:
                # 强教师分歧 → 用 Llama（若 Llama 高置信）否则用 dsv
                ans = llama[u][0] if llama[u][2] < th else dsv[u][0]
            if ans == dsv[u][1]:
                c += 1
        print(f"  θ<{th:4.2f}: {100.0*c/len(common):.2f}%", flush=True)

    # 规则C：分歧题直接用 Llama（不管置信度）
    print("\n=== 规则C: 强教师一致→用；分歧→直接用Llama（无阈值） ===", flush=True)
    c = 0
    for u in common:
        ans = dsv[u][0] if dsv[u][0] == dob[u][0] else llama[u][0]
        if ans == dsv[u][1]:
            c += 1
    print(f"  无阈值: {100.0*c/len(common):.2f}%", flush=True)


if __name__ == "__main__":
    main()
