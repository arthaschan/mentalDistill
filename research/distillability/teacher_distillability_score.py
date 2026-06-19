#!/usr/bin/env python3
"""
teacher_distillability_score.py — 教师级「可蒸馏性」先验预测器（阶段 2 预注册用）

科学约束：本脚本定义的所有分数都是【先验】的——只依赖教师真实 logprobs 的几何/
统计性质 + 阶段 1 已确立的理论，绝不拟合任何学生蒸馏增益。这样阶段 2 才能做
「预测在前、验证在后」的盲验证。

输入：教师真实 logprobs jsonl（含 TeacherDist；GT 读 OriginalAnswer or Answer）。
输出：每个教师一行打分 + 排序，存 JSON。

预测器（均为先验，无需训练学生）：
  teacher_acc          : 教师 argmax 准确率（论文倒 U 的横轴）
  disagree_rate        : 教师-GT 不一致率 = 1 - acc（暗知识/噪声的粗代理）
  geom_auc             : 用 GT-无关几何特征预测「教师该样本对错」的 5 折 CV AUC
                         （阶段 1 核心量：错误是否"几何可分"=可被 training-free 过滤）
  mean_logdet_g        : 平均 log10 体积元（分布尖锐度；真 logprobs vs 平滑的 2500x 差异源）
  correct_wrong_sep    : 答对组与答错组在 logdet_g 上的 |Cohen's d|（错误几何可分度）
  --- 复合先验预测分（阶段 2 预注册的主预测量）---
  distillability_index : 综合分。直觉：教师要"够强"(acc 高) 且其"错误可被几何识别"
                         (geom_auc 高、correct_wrong_sep 大)，才既能提供可迁移结构、
                         又能让低质量样本被几何过滤掉。
                         DI = z(teacher_acc) + z(geom_auc) + z(correct_wrong_sep)
                         （三者标准化后等权相加；阶段 2 跑完后用真实增益检验其排序效力）

用法：
  python teacher_distillability_score.py \
    --teachers Qwen14B:path1 Qwen32B:path2 Llama70B:path3 \
    --output research/distillability/outputs/teacher_distillability_scores.json
"""
import argparse
import json
import math
import os
import random

OPTION_LETTERS = ["A", "B", "C", "D", "E"]
GT_INDEP_FEATS = ["logdet_g", "boundary", "entropy", "margin", "peak"]


def parse_dist(row):
    dist = row.get("TeacherDist", {})
    gt = row.get("OriginalAnswer") or row.get("Answer", "")
    gt = str(gt).strip().upper()
    if not dist or gt not in OPTION_LETTERS:
        return None, None, None
    raw = [float(dist.get(ch, 0.0)) for ch in OPTION_LETTERS]
    if sum(raw) <= 1e-9:
        return None, None, None
    probs = [max(v, 1e-12) for v in raw]
    s = sum(probs)
    probs = [p / s for p in probs]
    teacher_ans = OPTION_LETTERS[raw.index(max(raw))]
    return probs, gt, teacher_ans


def feats(p):
    srt = sorted(p, reverse=True)
    return {
        "logdet_g": -0.5 * sum(math.log10(x) for x in p),
        "boundary": min(p),
        "entropy": -sum(x * math.log(x + 1e-12) for x in p),
        "margin": srt[0] - srt[1],
        "peak": srt[0],
    }


def load(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p, gt, ta = parse_dist(json.loads(line))
            if p is None:
                continue
            fr = feats(p)
            fr["correct"] = 1 if ta == gt else 0
            rows.append(fr)
    return rows


def cohens_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    ma, mb = sum(a) / na, sum(b) / nb
    va = sum((x - ma) ** 2 for x in a) / (na - 1)
    vb = sum((x - mb) ** 2 for x in b) / (nb - 1)
    pooled = math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    return (ma - mb) / pooled if pooled else 0.0


def auc_roc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return 0.5
    pr = sorted(zip(scores, labels))
    rank_pos = 0.0
    i = 0
    N = len(pr)
    while i < N:
        j = i
        while j < N and pr[j][0] == pr[i][0]:
            j += 1
        ar = (i + 1 + j) / 2.0
        for k in range(i, j):
            if pr[k][1] == 1:
                rank_pos += ar
        i = j
    return (rank_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def train_logreg(X, y, epochs=400, lr=0.1, l2=1e-3):
    d = len(X[0])
    w = [0.0] * d
    b = 0.0
    m = len(y)
    for _ in range(epochs):
        gw = [0.0] * d
        gb = 0.0
        for xi, yi in zip(X, y):
            z = max(min(b + sum(wj * xj for wj, xj in zip(w, xi)), 30), -30)
            e = 1 / (1 + math.exp(-z)) - yi
            for k in range(d):
                gw[k] += e * xi[k]
            gb += e
        for k in range(d):
            w[k] -= lr * (gw[k] / m + l2 * w[k])
        b -= lr * gb / m
    return w, b


def geom_cv_auc(rows, folds=5, seed=0):
    y = [r["correct"] for r in rows]
    if len(set(y)) < 2:
        return None  # teacher all-correct or all-wrong on kept set
    X = [[r[f] for f in GT_INDEP_FEATS] for r in rows]
    n = len(y)
    cols = list(zip(*X))
    means = [sum(c) / n for c in cols]
    stds = [(sum((v - mu) ** 2 for v in c) / n) ** 0.5 or 1.0 for c, mu in zip(cols, means)]
    Xs = [[(v - mu) / s for v, mu, s in zip(row, means, stds)] for row in X]
    random.seed(seed)
    idx = list(range(n))
    random.shuffle(idx)
    fold = [idx[i::folds] for i in range(folds)]
    sc, lb = [], []
    for fi in range(folds):
        te = set(fold[fi])
        tr = [i for i in range(n) if i not in te]
        w, b = train_logreg([Xs[i] for i in tr], [y[i] for i in tr])
        for i in fold[fi]:
            sc.append(b + sum(wj * xj for wj, xj in zip(w, Xs[i])))
            lb.append(y[i])
    return auc_roc(sc, lb)


def analyze(label, rows):
    n = len(rows)
    n_correct = sum(r["correct"] for r in rows)
    acc = 100.0 * n_correct / n if n else 0.0
    c = [r["logdet_g"] for r in rows if r["correct"] == 1]
    w = [r["logdet_g"] for r in rows if r["correct"] == 0]
    gauc = geom_cv_auc(rows)
    return {
        "label": label,
        "n_samples": n,
        "teacher_acc": round(acc, 2),
        "disagree_rate": round(100.0 - acc, 2),
        "geom_auc": round(gauc, 4) if gauc is not None else None,
        "mean_logdet_g": round(sum(r["logdet_g"] for r in rows) / n, 4) if n else None,
        "correct_wrong_sep": round(abs(cohens_d(c, w)), 4),
    }


def zscores(vals):
    vals = [v for v in vals if v is not None]
    if len(vals) < 2:
        return None
    mu = sum(vals) / len(vals)
    sd = (sum((v - mu) ** 2 for v in vals) / len(vals)) ** 0.5 or 1.0
    return mu, sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teachers", nargs="+", required=True, help="label:path pairs")
    ap.add_argument("--output", default="research/distillability/outputs/teacher_distillability_scores.json")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    analyses = []
    for spec in args.teachers:
        label, path = spec.split(":", 1)
        if not os.path.exists(path):
            print(f"[SKIP] {label}: {path} not found")
            continue
        rows = load(path)
        analyses.append(analyze(label, rows))

    # composite distillability index = z(acc) + z(geom_auc) + z(correct_wrong_sep)
    accs = [a["teacher_acc"] for a in analyses]
    aucs = [a["geom_auc"] for a in analyses]
    seps = [a["correct_wrong_sep"] for a in analyses]
    za, zauc, zsep = zscores(accs), zscores(aucs), zscores(seps)
    for a in analyses:
        di = 0.0
        if za:
            di += (a["teacher_acc"] - za[0]) / za[1]
        if zauc and a["geom_auc"] is not None:
            di += (a["geom_auc"] - zauc[0]) / zauc[1]
        if zsep:
            di += (a["correct_wrong_sep"] - zsep[0]) / zsep[1]
        a["distillability_index"] = round(di, 4)

    ranked = sorted(analyses, key=lambda x: -x["distillability_index"])
    for rank, a in enumerate(ranked, 1):
        a["predicted_rank"] = rank

    report = {
        "note": "PRIOR distillability predictors. Computed from teacher logprobs ONLY; "
                "no student-distillation outcome was used. Freeze (git commit) before any "
                "Phase-2 student training to enable prospective validation.",
        "teachers": ranked,
    }
    json.dump(report, open(args.output, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print("=" * 92)
    print("Teacher-level PRIOR distillability predictors (no student outcome used)")
    print("=" * 92)
    print(f"{'rank':>4} {'teacher':<14}{'acc%':>7}{'geom_auc':>10}{'logdet_g':>10}"
          f"{'corr/wrong_sep':>15}{'DI':>9}")
    for a in ranked:
        print(f"{a['predicted_rank']:>4} {a['label']:<14}{a['teacher_acc']:>7.2f}"
              f"{(a['geom_auc'] or 0):>10.4f}{(a['mean_logdet_g'] or 0):>10.4f}"
              f"{a['correct_wrong_sep']:>15.4f}{a['distillability_index']:>9.4f}")
    print(f"\n[SAVED] {args.output}")


if __name__ == "__main__":
    main()
