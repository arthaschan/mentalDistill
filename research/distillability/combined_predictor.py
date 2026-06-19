#!/usr/bin/env python3
"""
combined_predictor.py — 方向 B 阶段 1 的多特征联合判别力检验

在样本级几何特征 CSV 上，检验「GT 无关」的置信度几何量
（logdet_g, boundary, entropy, margin, peak）联合起来，能否在
不看 GT 的前提下预测「教师该样本是否答对」。

为什么排除 fr_to_gt：它直接用 GT 计算（d=2·arccos(√p_gt)），
对「预测教师对错」是同义反复（AUC≈1），不能作为 training-free 信号。
真正有研究价值的是：仅凭分布形状（置信度几何）能否预判可蒸馏性。

方法：5 折交叉验证 + 手写逻辑回归（零依赖，纯 stdlib）。
输出标准化权重（可解释哪个几何量最重要）和 CV AUC。
"""
import argparse
import csv
import json
import math
import os
import random

GT_INDEPENDENT = ["logdet_g", "boundary", "entropy", "margin", "peak"]


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


def train_logreg(X, y, epochs=300, lr=0.1):
    d = len(X[0])
    w = [0.0] * d
    b = 0.0
    m = len(y)
    for _ in range(epochs):
        gw = [0.0] * d
        gb = 0.0
        for xi, yi in zip(X, y):
            z = b + sum(wj * xj for wj, xj in zip(w, xi))
            z = max(min(z, 30), -30)
            p = 1 / (1 + math.exp(-z))
            e = p - yi
            for k in range(d):
                gw[k] += e * xi[k]
            gb += e
        for k in range(d):
            w[k] -= lr * gw[k] / m
        b -= lr * gb / m
    return w, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="per-sample geometry CSV from sample_geometry.py")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="research/distillability/outputs/combined_predictor_report.json")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    feats = GT_INDEPENDENT
    X = [[float(r[f]) for f in feats] for r in rows]
    y = [int(r["correct"]) for r in rows]
    n = len(y)

    # standardize
    cols = list(zip(*X))
    means = [sum(c) / n for c in cols]
    stds = [(sum((v - m) ** 2 for v in c) / n) ** 0.5 or 1.0 for c, m in zip(cols, means)]
    Xs = [[(v - m) / s for v, m, s in zip(row, means, stds)] for row in X]

    random.seed(args.seed)
    idx = list(range(n))
    random.shuffle(idx)
    folds = [idx[i::args.folds] for i in range(args.folds)]
    cv_scores, cv_labels = [], []
    for fi in range(args.folds):
        te = set(folds[fi])
        tr = [i for i in range(n) if i not in te]
        w, b = train_logreg([Xs[i] for i in tr], [y[i] for i in tr])
        for i in folds[fi]:
            z = b + sum(wj * xj for wj, xj in zip(w, Xs[i]))
            cv_scores.append(z)
            cv_labels.append(y[i])
    cv_auc = auc_roc(cv_scores, cv_labels)

    w_full, b_full = train_logreg(Xs, y)
    single = {}
    for k, f in enumerate(feats):
        a = auc_roc([row[k] for row in X], y)
        single[f] = round(max(a, 1 - a), 4)

    report = {
        "csv": args.csv,
        "n_samples": n,
        "n_correct": sum(y),
        "teacher_acc": round(100.0 * sum(y) / n, 2),
        "combined_cv_auc": round(cv_auc, 4),
        "standardized_weights": {f: round(wj, 4) for f, wj in zip(feats, w_full)},
        "single_feature_auc": single,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    json.dump(report, open(args.output, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print("=" * 70)
    print("Combined GT-independent distillability predictor")
    print("=" * 70)
    print(f"  samples={n}  teacher_acc={report['teacher_acc']}%")
    print(f"  combined {args.folds}-fold CV AUC = {cv_auc:.4f}")
    print(f"  standardized weights: {report['standardized_weights']}")
    print("  single-feature AUC (|·|, GT-independent):")
    for f, a in sorted(single.items(), key=lambda kv: -kv[1]):
        print(f"    {f:<10} {a:.4f}")
    print(f"\n[SAVED] {args.output}")


if __name__ == "__main__":
    main()
