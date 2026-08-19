#!/usr/bin/env python3
"""提取共识错题：dsv4pro 和 doubao 都答错的题（两个老师共享的盲点）。

输出 consensus_errors.jsonl：每行含 uid/Question/Options/GT + 两老师答案。
"""
import json

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
            d[uid] = (ta, gt, r)
    return d


def main():
    dsv = load("26_collaborative_distill/data/labels/dsv4pro_test_full.jsonl")
    dob = load("26_collaborative_distill/data/labels/doubao_turbo_test_full.jsonl")
    common = set(dsv) & set(dob)

    out = []
    both_wrong = 0
    both_wrong_same = 0
    for uid in sorted(common):
        a, gt, ra = dsv[uid]
        b, _, _ = dob[uid]
        if a != gt and b != gt:
            both_wrong += 1
            if a == b:
                both_wrong_same += 1
            out.append({
                "uid": uid,
                "Question": ra.get("Question", ""),
                "Options": ra.get("Options", ""),
                "GT": gt,
                "dsv4pro": a,
                "doubao": b,
                "same_wrong": (a == b),
            })

    with open("26_collaborative_distill/data/consensus_errors.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"交集 {len(common)} 题, 共识错题(都错) {both_wrong} 题, "
          f"其中错同一个答案 {both_wrong_same} 题")
    print(f"-> 26_collaborative_distill/data/consensus_errors.jsonl")


if __name__ == "__main__":
    main()
