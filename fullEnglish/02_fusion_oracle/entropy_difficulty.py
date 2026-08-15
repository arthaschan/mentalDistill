#!/usr/bin/env python3
"""fullEnglish — 熵=难度外部验证 (moat, 零 GPU).

英文医学无人类难度标注 -> 用「跨模型共识错误数」作难度金标准.
验证 (对齐英文牙科 H4/5d/5d-null):
  H4      教师高熵(低置信)子集错误率显著高于低熵子集
  5d      教师熵 与 跨模型共识难度 相关 (外部金标准)
  5d-null 熵不归约到表面文本 artifact (题长/否定词数)
输出 fullEnglish/02_fusion_oracle/entropy_difficulty.{json,md}
"""
import json
import os
import glob
import re
import numpy as np

FE = os.path.dirname(os.path.abspath(__file__))
LP = os.path.join(os.path.dirname(FE), "01_teacher_screening", "logprobs")
LETTERS = ["A", "B", "C", "D", "E"]


def spearman(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ar = np.argsort(np.argsort(a))
    br = np.argsort(np.argsort(b))
    if ar.std() == 0 or br.std() == 0:
        return 0.0
    return float(np.corrcoef(ar, br)[0, 1])


def perm_pvalue(a, b, rho, iters=1000, seed=0):
    rng = np.random.RandomState(seed)
    b = np.asarray(b, float)
    cnt = 0
    for _ in range(iters):
        if abs(spearman(a, rng.permutation(b))) >= abs(rho):
            cnt += 1
    return (cnt + 1) / (iters + 1)


def load(path):
    out = {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        dist = r.get("TeacherDist", {})
        gt = str(r.get("OriginalAnswer") or r.get("Answer", "")).strip().upper()
        uid = r.get("uid")
        if not dist or gt not in LETTERS or not uid:
            continue
        raw = np.array([float(dist.get(c, 0.0)) for c in LETTERS])
        if raw.sum() <= 1e-9:
            continue
        raw = raw / raw.sum()
        ent = float(-np.sum(np.clip(raw, 1e-12, None) * np.log(np.clip(raw, 1e-12, None))))
        pred = LETTERS[int(np.argmax(raw))]
        out[uid] = {"gt": gt, "ent": ent, "pred": pred, "correct": int(pred == gt),
                    "src": r.get("source", "?"), "stem": r.get("Question", "")}
    return out


teachers = {}
for f in sorted(glob.glob(os.path.join(LP, "*_logprobs.jsonl"))):
    name = os.path.basename(f).replace("_logprobs.jsonl", "")
    d = load(f)
    if d:
        teachers[name] = d
names = list(teachers)
common = sorted(set.intersection(*[set(d) for d in teachers.values()]))
N = len(common)

consensus_wrong = {u: sum(1 - teachers[n][u]["correct"] for n in names) for u in common}
report = {"n_common": N, "n_teachers": len(names), "teachers": names}

# H4
h4 = {}
for n in names:
    ents = np.array([teachers[n][u]["ent"] for u in common])
    errs = np.array([1 - teachers[n][u]["correct"] for u in common])
    thr = np.quantile(ents, 0.5)
    hi = ents > thr
    lo = ents <= thr
    hi_err = 100 * errs[hi].mean() if hi.sum() else None
    lo_err = 100 * errs[lo].mean() if lo.sum() else None
    ratio = (hi_err / lo_err) if (lo_err and lo_err > 0) else None
    h4[n] = {"acc": round(100 * (1 - errs.mean()), 2),
             "high_entropy_err": round(hi_err, 2) if hi_err is not None else None,
             "low_entropy_err": round(lo_err, 2) if lo_err is not None else None,
             "err_ratio": round(ratio, 2) if ratio else None}
report["H4_entropy_locates_errors"] = h4

# 5d
cons = [consensus_wrong[u] for u in common]
mean_ent = [np.mean([teachers[n][u]["ent"] for n in names]) for u in common]
rho_agg = spearman(mean_ent, cons)
five_d = {}
for n in names:
    ents = [teachers[n][u]["ent"] for u in common]
    five_d[n] = {"rho": round(spearman(ents, cons), 4),
                 "p_perm": round(perm_pvalue(ents, cons, spearman(ents, cons), seed=1), 5)}
report["5d_entropy_vs_consensus"] = {
    "per_teacher": five_d,
    "mean_entropy_vs_consensus_rho": round(rho_agg, 4),
    "p_perm": round(perm_pvalue(mean_ent, cons, rho_agg, seed=2), 5),
}

grad = {}
for w in range(len(names) + 1):
    us = [u for u in common if consensus_wrong[u] == w]
    if us:
        grad[w] = {"n": len(us),
                   "mean_teacher_ent": round(float(np.mean([mean_ent[common.index(u)] for u in us])), 4)}
report["consensus_gradient"] = grad

# 5d-null
NEG = re.compile(r'\b(not|except|least|never|cannot|false|incorrect|unlikely|contraindicated)\b', re.I)


def stem_len(u):
    return len(teachers[names[0]][u]["stem"].split())


def neg_count(u):
    return len(NEG.findall(teachers[names[0]][u]["stem"]))


slen = [stem_len(u) for u in common]
negc = [neg_count(u) for u in common]
report["5d_null_surface"] = {
    "mean_entropy_vs_stem_length_rho": round(spearman(mean_ent, slen), 4),
    "mean_entropy_vs_negation_count_rho": round(spearman(mean_ent, negc), 4),
    "consensus_vs_stem_length_rho": round(spearman(cons, slen), 4),
}

json.dump(report, open(os.path.join(FE, "entropy_difficulty.json"), "w"), ensure_ascii=False, indent=2)

md = [f"# fullEnglish — 熵=难度外部验证 (n={N}, {len(names)} teachers)\n",
      "金标准 = 跨模型共识错误数 (英文医学无人类难度标注).\n",
      "## H4: 熵定位教师自身的错误子集",
      "| teacher | acc% | high-entropy err% | low-entropy err% | ratio |", "|---|---|---|---|---|"]
for n in sorted(h4, key=lambda k: -h4[k]["acc"]):
    v = h4[n]
    md.append(f"| {n} | {v['acc']} | {v['high_entropy_err']} | {v['low_entropy_err']} | {v['err_ratio']}x |")
md.append("\n## 5d: 熵 vs 跨模型共识难度 (外部金标准)")
md.append(f"- **mean-entropy vs consensus rho = {report['5d_entropy_vs_consensus']['mean_entropy_vs_consensus_rho']}** (p_perm={report['5d_entropy_vs_consensus']['p_perm']})")
md.append("\n| teacher | rho | p_perm |")
md.append("|---|---|---|")
for n in names:
    v = five_d[n]
    md.append(f"| {n} | {v['rho']} | {v['p_perm']} |")
md.append("\n### 共识难度梯度 (教师平均熵随错题数上升)")
md.append("| #teachers wrong | n items | mean teacher entropy |")
md.append("|---|---|---|")
for w, v in grad.items():
    md.append(f"| {w} | {v['n']} | {v['mean_teacher_ent']} |")
md.append("\n## 5d-null: 表面文本 artifact 对照 (期望约 0)")
s = report["5d_null_surface"]
md.append(f"- entropy vs stem length rho = {s['mean_entropy_vs_stem_length_rho']}")
md.append(f"- entropy vs #negation words rho = {s['mean_entropy_vs_negation_count_rho']}")
md.append(f"- consensus vs stem length rho = {s['consensus_vs_stem_length_rho']}")
open(os.path.join(FE, "entropy_difficulty.md"), "w").write("\n".join(md))

print(f"=== 熵=难度 (n={N}, {len(names)} teachers) ===")
for n in sorted(h4, key=lambda k: -h4[k]["acc"]):
    print(f"  {n:14s} acc={h4[n]['acc']}%  hi-ent-err={h4[n]['high_entropy_err']}% vs lo={h4[n]['low_entropy_err']}%  ({h4[n]['err_ratio']}x)")
print(f"5d mean-entropy vs consensus: rho={report['5d_entropy_vs_consensus']['mean_entropy_vs_consensus_rho']} p={report['5d_entropy_vs_consensus']['p_perm']}")
print(f"5d-null: len rho={s['mean_entropy_vs_stem_length_rho']}  neg rho={s['mean_entropy_vs_negation_count_rho']}")
print(f"-> {FE}/entropy_difficulty.md")
