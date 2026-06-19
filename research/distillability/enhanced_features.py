#!/usr/bin/env python3
"""
enhanced_features.py — 方向 B / 任务 C：增强的「混淆结构」几何特征

阶段 1 用 5 个置信度几何量（熵/峰值/margin/体积元/边界）达到 AUC≈0.66。
天花板的原因：它们都在刻画同一件事——「分布有多尖」。

任务 C 的假设（来自论文「有价值的犹豫 vs 噪声摇摆」）：
  教师答对常伴随「两个选项之间的二元犹豫」（质量集中在 top-2）；
  教师答错常伴随「质量散布到多个选项」（接近均匀的噪声）。
单纯的「尖锐度」无法区分「二元犹豫」与「单峰自信」，需要刻画
**分布的有效支撑数 / 模态结构**的 GT-无关特征：

  - participation_ratio : (Σp)²/Σp² = 1/Σp²，有效选项数（IPR 倒数）
  - renyi2              : -log Σp²（碰撞熵，对尾部更敏感）
  - top2_mass          : 前两选项概率和（二元犹豫时接近 1）
  - top3_minus_top2    : 第三选项的额外质量（噪声散布的标志）
  - top2_ratio         : p(2)/p(1)，犹豫的强度
  - gini               : 概率的基尼系数（不平等度）
  - tsallis2           : Tsallis q=2 熵 = 1-Σp²
  - second_concentration: p(2)/(1-p(1))，剩余质量是否集中在单一竞争者
                          （高=二元犹豫；低=多向噪声）

对每个特征：报告 correct/wrong 的 Cohen's d、单特征 AUC；
再做两个联合预测器的 5 折 CV AUC 对比：
  (1) baseline = 阶段 1 的 5 个置信度特征
  (2) enhanced = baseline + 上述结构特征
看结构特征能否把 AUC 推过 0.70。
"""
import argparse
import csv
import json
import math
import os
import random

OPTION_LETTERS = ["A", "B", "C", "D", "E"]

BASELINE_FEATS = ["logdet_g", "boundary", "entropy", "margin", "peak"]
STRUCT_FEATS = ["participation_ratio", "renyi2", "top2_mass", "top3_minus_top2",
                "top2_ratio", "gini", "tsallis2", "second_concentration"]


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


def features(p, gt, teacher_ans):
    srt = sorted(p, reverse=True)
    sq = sum(x * x for x in p)
    H = -sum(x * math.log(x + 1e-12) for x in p)
    # baseline (sharpness) features
    feat = {
        "logdet_g": -0.5 * sum(math.log10(x) for x in p),
        "boundary": min(p),
        "entropy": H,
        "margin": srt[0] - srt[1],
        "peak": srt[0],
    }
    # structural (mode/support) features — all GT-independent
    feat["participation_ratio"] = 1.0 / sq                       # effective #options in [1,5]
    feat["renyi2"] = -math.log(sq)                               # collision entropy
    feat["top2_mass"] = srt[0] + srt[1]
    feat["top3_minus_top2"] = srt[2]                             # mass leaking to 3rd option
    feat["top2_ratio"] = srt[1] / (srt[0] + 1e-12)
    mean = 1.0 / len(p)
    feat["gini"] = sum(abs(a - b) for a in p for b in p) / (2 * len(p) * len(p) * mean)
    feat["tsallis2"] = 1.0 - sq
    feat["second_concentration"] = srt[1] / (1.0 - srt[0] + 1e-12)  # is residual mass on ONE rival?
    feat["correct"] = 1 if teacher_ans == gt else 0
    return feat


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
            rows.append(features(p, gt, ta))
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


def cv_auc(X, y, folds=5, seed=0):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True, help="path to real-logprob teacher jsonl")
    ap.add_argument("--label", default="teacher")
    ap.add_argument("--outdir", default="research/distillability/outputs")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rows = load(args.teacher)
    y = [r["correct"] for r in rows]
    allfeats = BASELINE_FEATS + STRUCT_FEATS

    csv_path = os.path.join(args.outdir, f"enhanced_features_{args.label}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=allfeats + ["correct"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    correct = [r for r in rows if r["correct"] == 1]
    wrong = [r for r in rows if r["correct"] == 0]

    print("=" * 80)
    print(f"Enhanced confusion-structure features — {args.label}")
    print(f"n={len(rows)}  correct={len(correct)}  wrong={len(wrong)}  "
          f"teacher_acc={100*len(correct)/len(rows):.2f}%")
    print("=" * 80)
    print(f"{'feature':<22}{'correct_m':>11}{'wrong_m':>10}{'cohens_d':>10}{'auc':>8}")
    per_feat = {}
    for ft in allfeats:
        c = [r[ft] for r in correct]
        w_ = [r[ft] for r in wrong]
        vals = [r[ft] for r in rows]
        a = auc_roc(vals, y)
        d = cohens_d(c, w_)
        per_feat[ft] = {"cohens_d": round(d, 4), "auc": round(max(a, 1 - a), 4),
                        "correct_mean": round(sum(c) / len(c), 4),
                        "wrong_mean": round(sum(w_) / len(w_), 4)}
        tag = "*" if ft in STRUCT_FEATS else " "
        print(f"{tag}{ft:<21}{per_feat[ft]['correct_mean']:>11.4f}"
              f"{per_feat[ft]['wrong_mean']:>10.4f}{d:>10.4f}{max(a,1-a):>8.4f}")

    Xb = [[r[ft] for ft in BASELINE_FEATS] for r in rows]
    Xe = [[r[ft] for ft in allfeats] for r in rows]
    Xs = [[r[ft] for ft in STRUCT_FEATS] for r in rows]
    auc_b = cv_auc(Xb, y)
    auc_e = cv_auc(Xe, y)
    auc_s = cv_auc(Xs, y)

    print("\n" + "-" * 80)
    print("5-fold CV AUC (combined predictors):")
    print(f"  baseline (sharpness, 5 feats)        : {auc_b:.4f}")
    print(f"  structural only ({len(STRUCT_FEATS)} feats)            : {auc_s:.4f}")
    print(f"  enhanced (baseline + structural)     : {auc_e:.4f}")
    print(f"  delta (enhanced - baseline)          : {auc_e - auc_b:+.4f}")

    report = {
        "label": args.label,
        "n": len(rows), "n_correct": len(correct), "n_wrong": len(wrong),
        "teacher_acc": round(100 * len(correct) / len(rows), 2),
        "per_feature": per_feat,
        "cv_auc": {"baseline": round(auc_b, 4), "structural_only": round(auc_s, 4),
                   "enhanced": round(auc_e, 4), "delta": round(auc_e - auc_b, 4)},
    }
    out_json = os.path.join(args.outdir, f"enhanced_features_report_{args.label}.json")
    json.dump(report, open(out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n[SAVED] {csv_path}")
    print(f"[SAVED] {out_json}")


if __name__ == "__main__":
    main()
