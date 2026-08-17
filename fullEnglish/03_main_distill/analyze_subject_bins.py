#!/usr/bin/env python3
"""按"教师强弱"分档做机制检验：合并 MMLU + MedMCQA 全部学科，看学生 Δ 是否随教师变弱而由负转正。"""
import json

RUN = "fullEnglish/03_main_distill/runs"


def load(p):
    with open(p) as f:
        return json.load(f)


mmlu = load(f"{RUN}/mmlu_subject_analysis.json")
med = load(f"{RUN}/medmcqa_subject_analysis.json")

# 合并 (subject, n, teacher_acc, student_mean, delta)
subs = []
for d in (mmlu, med):
    for k, v in d.items():
        if v.get("n") and v["n"] >= 10 and v.get("teacher_acc") is not None:
            subs.append((k, v["n"], v["teacher_acc"], v["student_mean"], v["delta"]))

print(f"合并学科数: {len(subs)}（n>=10）")

# 分档：按教师正确率
bins = [("教师<70%", 0, 70), ("70-80%", 70, 80), ("80-90%", 80, 90), ("教师>90%", 90, 101)]
print(f"\n{'档':14s} {'学科数':>6s} {'总题数':>7s} {'教师均':>8s} {'学生均':>8s} {'Δ加权':>8s}")
for name, lo, hi in bins:
    grp = [s for s in subs if lo <= s[2] < hi]
    if not grp:
        continue
    n_tot = sum(s[1] for s in grp)
    t_w = sum(s[2] * s[1] for s in grp) / n_tot
    st_w = sum(s[3] * s[1] for s in grp) / n_tot
    d_w = sum(s[4] * s[1] for s in grp) / n_tot
    print(f"{name:14s} {len(grp):6d} {n_tot:7d} {t_w:8.2f} {st_w:8.2f} {d_w:>+8.2f}")

# 详细：每档里超越/接近的学科
print("\n=== 各档学科明细（按 Δ 排序）===")
for name, lo, hi in bins:
    grp = [s for s in subs if lo <= s[2] < hi]
    if not grp:
        continue
    print(f"\n[{name}]")
    for s in sorted(grp, key=lambda x: -x[4]):
        flag = " <<<超越" if s[4] > 0 else ""
        print(f"  {s[0]:30s} n={s[1]:5d} 教师={s[2]:6.2f} 学生={s[3]:6.2f} Δ={s[4]:+7.2f}{flag}")
