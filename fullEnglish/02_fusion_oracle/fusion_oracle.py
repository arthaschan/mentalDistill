#!/usr/bin/env python3
"""fullEnglish — 多教师融合上界 (GO/NO-GO 闸门, 零训练, 可选步骤 02).

复用 01_teacher_screening 的标签 (labels/ = API 硬标签, logprobs/ = 本地真实分布).
回答: 多个教师 (DeepSeek/Gemini/Llama/...) 之间是否存在可廉价捕获的互补,
值得走「多教师融合蒸馏」, 还是走「单最强教师蒸馏」.

判据 (预注册, 与中文/英文牙科一致):
  achievable ceiling - best_single >= 2.0pp -> GO   (融合)
                                  < 0.5pp -> NO-GO (单教师)
  其余 -> WEAK-GO (记录, 二级实验)

策略:
  best_single      最强单教师 (基线)
  oracle_anyright  任一教师答对即对 (松上界, 用 GT 不可实现)
  majority_vote    硬标签多数投票 (label-free, 可实现)
  domain_route_CV  每源最强教师路由, 5 折交叉验证 (label-free, 诚实可实现)
  prob_avg         概率平均 (仅本地真实分布教师参与)
  conf_route       最低熵教师路由 (仅本地真实分布教师参与)
"""
import json
import os
import glob
import numpy as np
from collections import Counter, defaultdict

FE = os.path.dirname(os.path.abspath(__file__))
SCREEN = os.path.join(os.path.dirname(FE), "01_teacher_screening")
LETTERS = ["A", "B", "C", "D", "E"]


def load_rows(path):
    rows = {}
    if not os.path.exists(path):
        return rows
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        gt = str(r.get("OriginalAnswer") or r.get("Answer", "")).strip().upper()
        pred = str(r.get("TeacherAnswer") or r.get("Answer", "")).strip().upper()
        if gt not in LETTERS or pred not in LETTERS:
            continue
        uid = r.get("uid") or r.get("Question", "")[:40]
        dist = r.get("TeacherDist", {})
        ent = None
        if isinstance(dist, dict) and dist:
            raw = np.array([float(dist.get(c, 0.0)) for c in LETTERS])
            s = raw.sum()
            if s > 1e-9:
                raw = raw / s
                ent = float(-np.sum(np.clip(raw, 1e-12, None) * np.log(np.clip(raw, 1e-12, None))))
        rows[uid] = {"gt": gt, "pred": pred, "src": r.get("source", "?"),
                     "subj": r.get("subject", "?"), "ent": ent, "dist": raw if ent is not None else None}
    return rows


def main():
    teachers = {}
    for f in sorted(glob.glob(os.path.join(SCREEN, "labels", "*.jsonl"))):
        d = load_rows(f)
        if d:
            teachers[os.path.basename(f).replace(".jsonl", "")] = d
    for f in sorted(glob.glob(os.path.join(SCREEN, "logprobs", "*_logprobs.jsonl"))):
        d = load_rows(f)
        if d:
            teachers[os.path.basename(f).replace("_logprobs.jsonl", "")] = d

    if len(teachers) < 2:
        print("[fusion oracle] 教师 <2 个, 跳过 (先跑 01_teacher_screening)")
        raise SystemExit(0)

    names = list(teachers)
    common = sorted(set.intersection(*[set(d) for d in teachers.values()]))
    N = len(common)

    overall = {n: 100 * np.mean([teachers[n][u]["pred"] == teachers[n][u]["gt"] for u in common]) for n in names}
    best_single = max(overall, key=overall.get)
    bs = overall[best_single]

    # per-source best teacher
    src_items = defaultdict(list)
    for u in common:
        src_items[teachers[names[0]][u]["src"]].append(u)
    src_best = {}
    for s, us in src_items.items():
        accs = {n: np.mean([teachers[n][u]["pred"] == teachers[n][u]["gt"] for u in us]) for n in names}
        src_best[s] = max(accs, key=accs.get)

    res = {"best_single": round(bs, 2)}
    res["oracle_anyright"] = round(100 * np.mean(
        [any(teachers[n][u]["pred"] == teachers[names[0]][u]["gt"] for n in names) for u in common]), 2)

    def majvote(u):
        votes = Counter(teachers[n][u]["pred"] for n in names)
        top = max(votes.values())
        cands = [k for k, v in votes.items() if v == top]
        if len(cands) == 1:
            return cands[0]
        probs = {}
        for c in cands:
            probs[c] = sum(teachers[n][u]["dist"][LETTERS.index(c)] for n in names if teachers[n][u]["dist"] is not None)
        return max(probs, key=probs.get) if any(probs.values()) else cands[0]

    res["majority_vote"] = round(100 * np.mean(
        [majvote(u) == teachers[names[0]][u]["gt"] for u in common]), 2)

    # prob_avg / conf_route: 仅真实分布教师参与
    dist_names = [n for n in names if all(teachers[n][u]["dist"] is not None for u in common)]
    if len(dist_names) >= 2:
        res["prob_avg"] = round(100 * np.mean([
            LETTERS[int(np.argmax(np.mean([teachers[n][u]["dist"] for n in dist_names], axis=0)))]
            == teachers[names[0]][u]["gt"] for u in common]), 2)
        res["conf_route"] = round(100 * np.mean([
            teachers[min(dist_names, key=lambda n: teachers[n][u]["ent"])][u]["pred"]
            == teachers[names[0]][u]["gt"] for u in common]), 2)

    # domain route CV (label-free, 诚实可实现)
    def domain_route_cv(k=5, seed=42):
        rng = np.random.RandomState(seed)
        idx = np.array(common)
        rng.shuffle(idx)
        folds = np.array_split(idx, k)
        correct = 0
        for i in range(k):
            test = set(folds[i].tolist())
            train = [u for u in common if u not in test]
            sb = {}
            si = defaultdict(list)
            for u in train:
                si[teachers[names[0]][u]["src"]].append(u)
            for s, us in si.items():
                accs = {n: np.mean([teachers[n][u]["pred"] == teachers[n][u]["gt"] for u in us]) for n in names}
                sb[s] = max(accs, key=accs.get)
            for u in folds[i]:
                if teachers[sb.get(teachers[names[0]][u]["src"], best_single)][u]["pred"] == teachers[names[0]][u]["gt"]:
                    correct += 1
        return 100 * correct / N

    res["domain_route_CV"] = round(domain_route_cv(), 2)

    achievable = max([res.get(k, -1) for k in ["majority_vote", "domain_route_CV", "prob_avg", "conf_route"]])
    gap = round(achievable - bs, 2)
    oracle_gap = round(res["domain_route_CV"] - bs, 2) if "domain_route_CV" in res else None
    verdict = "GO" if gap >= 2.0 else ("NO-GO" if gap < 0.5 else "WEAK-GO")

    out = {"n_common": N, "teachers": names, "overall_acc": {k: round(v, 2) for k, v in overall.items()},
           "best_single": best_single, "best_single_acc": round(bs, 2),
           "fusion": res, "achievable_ceiling": achievable, "gap_pp": gap,
           "verdict": verdict, "per_source_best": src_best}
    os.makedirs(os.path.join(FE), exist_ok=True)
    with open(os.path.join(FE, "fusion_oracle.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"=== FUSION ORACLE (n={N}, {len(names)} teachers) ===")
    for n in sorted(names, key=lambda x: -overall[x]):
        print(f"  {n:14s} {overall[n]:.2f}%")
    print(f"best single: {best_single} = {bs:.2f}%")
    for k in sorted(res):
        if k == "best_single":
            continue
        print(f"  {k:18s} {res[k]:.2f}%  ({res[k]-bs:+.2f}pp)")
    print(f"ACHIEVABLE ceiling: {achievable:.2f}%  gap={gap:+.2f}pp  ->  VERDICT: {verdict}")
    print(f"-> {FE}/fusion_oracle.json")


if __name__ == "__main__":
    main()
