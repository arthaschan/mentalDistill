#!/usr/bin/env python3
"""
sample_geometry.py — 样本级可蒸馏性几何分析（方向 B 阶段 1）

在 5-选项概率单纯形 Δ⁴ 上，对每个教师样本计算信息几何特征，
并按「教师是否答对 GT」分组，检验几何量能否区分
「有价值的犹豫（valuable hesitation）」与「噪声摇摆（noise）」。

核心几何量（Fisher information metric g_ij(p)=δ_ij/p_i 下）:
  - fr_to_gt   : Fisher-Rao 测地距离到 GT one-hot, d=2·arccos(√p_gt)
  - logdet_g   : log10 体积元 = -0.5·Σ log10 p_i  (det g = 1/∏p_i)
  - boundary   : min_i p_i, 到单纯形边界的接近度（小=贴边=高置信）
  - entropy    : Shannon 熵 H(p)
  - margin     : top1 - top2 概率间隔（决策边界清晰度）
  - peak       : max_i p_i

输出:
  - outputs/sample_geometry_<teacher>.csv  每样本特征表
  - outputs/sample_geometry_report.json    分组统计 + 判别性指标
  - 控制台对照表
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path

OPTION_LETTERS = ["A", "B", "C", "D", "E"]


def parse_dist(row):
    dist = row.get("TeacherDist", {})
    # GT: prefer OriginalAnswer (raw generation keeps true GT here and overwrites
    # Answer with the teacher's prediction); fall back to Answer for built datasets.
    gt = row.get("OriginalAnswer") or row.get("Answer", "")
    gt = str(gt).strip().upper()
    if not dist or gt not in OPTION_LETTERS:
        return None, None, None
    raw = [float(dist.get(ch, 0.0)) for ch in OPTION_LETTERS]
    # Skip placeholder rows whose distribution is all-zero (filtered / SelectiveSource).
    if sum(raw) <= 1e-9:
        return None, None, None
    probs = [max(v, 1e-12) for v in raw]
    s = sum(probs)
    probs = [p / s for p in probs]
    # Determine teacher's hard prediction from the *raw* distribution argmax,
    # not from a stored TeacherAnswer (which may be GT-anchored in preprocessing).
    teacher_ans = OPTION_LETTERS[raw.index(max(raw))]
    return probs, gt, teacher_ans


def entropy(p):
    return -sum(pi * math.log(pi + 1e-12) for pi in p)


def fr_to_gt(p, gt):
    idx = OPTION_LETTERS.index(gt)
    bc = math.sqrt(p[idx])
    bc = min(max(bc, -1.0), 1.0)
    return 2.0 * math.acos(bc)


def logdet_g(p):
    # det g = 1 / prod(p_i); log10 volume element = -0.5 * sum log10 p_i
    return -0.5 * sum(math.log10(pi) for pi in p)


def margin(p):
    s = sorted(p, reverse=True)
    return s[0] - s[1]


def sample_features(p, gt, teacher_ans):
    return {
        "fr_to_gt": fr_to_gt(p, gt),
        "logdet_g": logdet_g(p),
        "boundary": min(p),
        "entropy": entropy(p),
        "margin": margin(p),
        "peak": max(p),
        "correct": 1 if teacher_ans == gt else 0,
    }


def load(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            p, gt, ta = parse_dist(r)
            if p is None:
                continue
            rows.append(sample_features(p, gt, ta))
    return rows


def stats(vals):
    n = len(vals)
    if n == 0:
        return {"n": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    mu = sum(vals) / n
    var = sum((x - mu) ** 2 for x in vals) / n
    return {"n": n, "mean": mu, "std": math.sqrt(var), "min": min(vals), "max": max(vals)}


def cohens_d(a, b):
    """效应量：两组均值差 / 合并标准差。衡量几何量对 correct/wrong 的判别力。"""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    ma, mb = sum(a) / na, sum(b) / nb
    va = sum((x - ma) ** 2 for x in a) / (na - 1)
    vb = sum((x - mb) ** 2 for x in b) / (nb - 1)
    pooled = math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled == 0:
        return 0.0
    return (ma - mb) / pooled


def auc_roc(scores, labels):
    """以单个几何量作为打分，预测 correct(=1) 的 AUC（Mann-Whitney U 等价）。"""
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return 0.5
    # rank-based AUC
    paired = sorted(zip(scores, labels), key=lambda x: x[0])
    ranks = {}
    i = 0
    n = len(paired)
    rank_sum_pos = 0.0
    # assign average ranks for ties
    idx = 0
    while idx < n:
        j = idx
        while j < n and paired[j][0] == paired[idx][0]:
            j += 1
        avg_rank = (idx + 1 + j) / 2.0  # ranks idx+1..j averaged
        for k in range(idx, j):
            if paired[k][1] == 1:
                rank_sum_pos += avg_rank
        idx = j
    n_pos = len(pos)
    n_neg = len(neg)
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return auc


FEATURES = ["fr_to_gt", "logdet_g", "boundary", "entropy", "margin", "peak"]


def analyze_teacher(label, rows, outdir):
    # write per-sample csv
    csv_path = os.path.join(outdir, f"sample_geometry_{label}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FEATURES + ["correct"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    correct_rows = [r for r in rows if r["correct"] == 1]
    wrong_rows = [r for r in rows if r["correct"] == 0]

    result = {
        "label": label,
        "n_samples": len(rows),
        "n_correct": len(correct_rows),
        "n_wrong": len(wrong_rows),
        "teacher_acc": round(100.0 * len(correct_rows) / len(rows), 2) if rows else 0.0,
        "overall": {},
        "correct_vs_wrong": {},
    }
    for feat in FEATURES:
        vals = [r[feat] for r in rows]
        result["overall"][feat] = {k: round(v, 4) for k, v in stats(vals).items()}
        c = [r[feat] for r in correct_rows]
        w_ = [r[feat] for r in wrong_rows]
        result["correct_vs_wrong"][feat] = {
            "correct_mean": round(sum(c) / len(c), 4) if c else 0.0,
            "wrong_mean": round(sum(w_) / len(w_), 4) if w_ else 0.0,
            "cohens_d": round(cohens_d(c, w_), 4),
            "auc_correct": round(auc_roc(vals, [r["correct"] for r in rows]), 4),
        }
    return result, csv_path


def main():
    ap = argparse.ArgumentParser(description="Sample-level distillability geometry (Stage 1, CPU only)")
    ap.add_argument("--teachers", nargs="+", required=True,
                    help="label:path pairs, e.g. 'Llama70B:16_.../teacher_train.jsonl'")
    ap.add_argument("--outdir", default="research/distillability/outputs")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    report = {"teachers": []}

    print("=" * 78)
    print("Sample-level Distillability Geometry — Stage 1 (CPU)")
    print("=" * 78)

    for spec in args.teachers:
        label, path = spec.split(":", 1)
        if not os.path.exists(path):
            print(f"\n[SKIP] {label}: file not found: {path}")
            continue
        rows = load(path)
        if not rows:
            print(f"\n[SKIP] {label}: no valid samples")
            continue
        res, csv_path = analyze_teacher(label, rows, args.outdir)
        report["teachers"].append(res)

        print(f"\n--- {label}  (n={res['n_samples']}, teacher_acc={res['teacher_acc']}%, "
              f"correct={res['n_correct']}, wrong={res['n_wrong']}) ---")
        print(f"  {'feature':<10}{'correct_mean':>13}{'wrong_mean':>12}{'cohens_d':>10}{'auc':>8}")
        for feat in FEATURES:
            cw = res["correct_vs_wrong"][feat]
            print(f"  {feat:<10}{cw['correct_mean']:>13.4f}{cw['wrong_mean']:>12.4f}"
                  f"{cw['cohens_d']:>10.4f}{cw['auc_correct']:>8.4f}")
        print(f"  [CSV] {csv_path}")

    out_json = os.path.join(args.outdir, "sample_geometry_report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[SAVED] {out_json}")

    # cross-teacher summary: which geometric quantity best separates correct/wrong?
    print("\n" + "=" * 78)
    print("Discriminative power summary (|Cohen's d|, averaged across teachers)")
    print("=" * 78)
    agg = {feat: [] for feat in FEATURES}
    for t in report["teachers"]:
        for feat in FEATURES:
            agg[feat].append(abs(t["correct_vs_wrong"][feat]["cohens_d"]))
    ranked = sorted(agg.items(), key=lambda kv: -(sum(kv[1]) / len(kv[1]) if kv[1] else 0))
    for feat, ds in ranked:
        mean_d = sum(ds) / len(ds) if ds else 0.0
        print(f"  {feat:<10} mean|d| = {mean_d:.4f}   per-teacher: {[round(x,3) for x in ds]}")


if __name__ == "__main__":
    main()
