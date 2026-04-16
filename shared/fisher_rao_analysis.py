#!/usr/bin/env python3
"""
fisher_rao_analysis.py — 教师软标签的 Fisher-Rao 信息几何分析

在 5-选项概率单纯形 Δ⁴ 上，计算各教师标签的几何特征：
1. Fisher-Rao 距离（测地线距离）：d_FR(p, q) = 2 · arccos(Σ √(pᵢqᵢ))
2. Bhattacharyya 系数 BC(p, q) = Σ √(pᵢqᵢ)
3. 标签多样性：到 GT one-hot 向量的 FR 距离分布
4. 教师间距离矩阵
5. 假软标签 vs 真软标签的几何差异

输入：各模块的 teacher_train_soft.jsonl 或 train_head_distill.jsonl
输出：JSON 报告 + 控制台表格
"""
import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path

OPTION_LETTERS = ["A", "B", "C", "D", "E"]


def parse_dist(row):
    """从一条 JSONL 记录中提取教师分布和 GT。"""
    dist = row.get("TeacherDist", {})
    gt = row.get("Answer", "")
    if not dist or gt not in OPTION_LETTERS:
        return None, None
    probs = []
    for ch in OPTION_LETTERS:
        probs.append(max(float(dist.get(ch, 0.0)), 1e-12))
    s = sum(probs)
    probs = [p / s for p in probs]
    return probs, gt


def bhattacharyya_coeff(p, q):
    """BC(p, q) = Σ √(pᵢ · qᵢ)"""
    return sum(math.sqrt(pi * qi) for pi, qi in zip(p, q))


def fisher_rao_distance(p, q):
    """d_FR(p, q) = 2 · arccos(BC(p, q)), clamped for numerical safety."""
    bc = bhattacharyya_coeff(p, q)
    bc = min(bc, 1.0)
    bc = max(bc, -1.0)
    return 2.0 * math.acos(bc)


def gt_onehot(gt_letter):
    """GT 的 one-hot 向量（带微小 epsilon 防止 log(0)）。"""
    eps = 1e-12
    vec = [eps] * len(OPTION_LETTERS)
    idx = OPTION_LETTERS.index(gt_letter)
    vec[idx] = 1.0 - eps * (len(OPTION_LETTERS) - 1)
    return vec


def entropy(p):
    """Shannon 熵 H(p)"""
    return -sum(pi * math.log(pi + 1e-12) for pi in p)


def max_entropy():
    """均匀分布的熵（5选项）"""
    return math.log(len(OPTION_LETTERS))


def load_teacher_labels(path):
    """加载 teacher label JSONL 文件，返回 (dist_list, gt_list)。"""
    dists = []
    gts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            dist, gt = parse_dist(row)
            if dist is not None:
                dists.append(dist)
                gts.append(gt)
    return dists, gts


def analyze_single_teacher(dists, gts, label):
    """分析单个教师的标签几何特征。"""
    fr_to_gt = []
    entropies = []
    peak_probs = []
    correct_count = 0

    for dist, gt in zip(dists, gts):
        gt_vec = gt_onehot(gt)
        d = fisher_rao_distance(dist, gt_vec)
        fr_to_gt.append(d)
        entropies.append(entropy(dist))
        peak_probs.append(max(dist))

        pred = OPTION_LETTERS[dist.index(max(dist))]
        if pred == gt:
            correct_count += 1

    n = len(dists)
    acc = 100.0 * correct_count / n if n > 0 else 0.0
    mean_fr = sum(fr_to_gt) / n if n > 0 else 0.0
    std_fr = math.sqrt(sum((x - mean_fr) ** 2 for x in fr_to_gt) / n) if n > 0 else 0.0
    mean_H = sum(entropies) / n if n > 0 else 0.0
    mean_peak = sum(peak_probs) / n if n > 0 else 0.0

    return {
        "label": label,
        "n_samples": n,
        "accuracy": round(acc, 2),
        "fr_to_gt_mean": round(mean_fr, 4),
        "fr_to_gt_std": round(std_fr, 4),
        "fr_to_gt_max": round(max(fr_to_gt) if fr_to_gt else 0, 4),
        "fr_to_gt_min": round(min(fr_to_gt) if fr_to_gt else 0, 4),
        "entropy_mean": round(mean_H, 4),
        "entropy_ratio": round(mean_H / max_entropy(), 4),  # 相对于均匀分布的熵比
        "peak_prob_mean": round(mean_peak, 4),
    }


def pairwise_teacher_distance(teachers_data):
    """计算教师间的平均 Fisher-Rao 距离矩阵。"""
    labels = list(teachers_data.keys())
    n_teachers = len(labels)
    dist_matrix = {}

    for i in range(n_teachers):
        for j in range(i + 1, n_teachers):
            li, lj = labels[i], labels[j]
            dists_i, gts_i = teachers_data[li]
            dists_j, gts_j = teachers_data[lj]

            # 取两个教师共有的样本（按位置对齐）
            n = min(len(dists_i), len(dists_j))
            fr_vals = []
            for k in range(n):
                d = fisher_rao_distance(dists_i[k], dists_j[k])
                fr_vals.append(d)

            mean_d = sum(fr_vals) / len(fr_vals) if fr_vals else 0.0
            dist_matrix[f"{li} ↔ {lj}"] = round(mean_d, 4)

    return dist_matrix


def fake_vs_real_analysis(fake_dists, real_dists, gts):
    """对比假软标签（smooth_eps）和真实软标签的几何差异。"""
    n = min(len(fake_dists), len(real_dists))
    fr_fake_gt = []
    fr_real_gt = []
    fr_fake_real = []
    entropy_fake = []
    entropy_real = []

    for k in range(n):
        gt_vec = gt_onehot(gts[k])
        fr_fake_gt.append(fisher_rao_distance(fake_dists[k], gt_vec))
        fr_real_gt.append(fisher_rao_distance(real_dists[k], gt_vec))
        fr_fake_real.append(fisher_rao_distance(fake_dists[k], real_dists[k]))
        entropy_fake.append(entropy(fake_dists[k]))
        entropy_real.append(entropy(real_dists[k]))

    return {
        "n_compared": n,
        "fake_fr_to_gt_mean": round(sum(fr_fake_gt) / n, 4) if n else 0,
        "real_fr_to_gt_mean": round(sum(fr_real_gt) / n, 4) if n else 0,
        "fake_real_fr_mean": round(sum(fr_fake_real) / n, 4) if n else 0,
        "fake_entropy_mean": round(sum(entropy_fake) / n, 4) if n else 0,
        "real_entropy_mean": round(sum(entropy_real) / n, 4) if n else 0,
        "entropy_ratio_fake_real": round(
            (sum(entropy_real) / n) / (sum(entropy_fake) / n), 4
        ) if n and sum(entropy_fake) > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Fisher-Rao information geometry analysis of teacher labels")
    parser.add_argument("--teachers", nargs="+", required=True,
                        help="label:path pairs, e.g. 'DeepSeek:02/data/teacher_train_soft.jsonl'")
    parser.add_argument("--fake_soft", type=str, default="",
                        help="Fake smooth_eps soft label file for comparison")
    parser.add_argument("--real_soft", type=str, default="",
                        help="Real multivote soft label file for comparison")
    parser.add_argument("--output", type=str, default="outputs/fisher_rao_report.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)

    report = {"analyses": [], "pairwise_distances": {}, "fake_vs_real": {}}
    teachers_data = {}

    print("=" * 70)
    print("Fisher-Rao Information Geometry Analysis")
    print("=" * 70)

    # Per-teacher analysis
    for spec in args.teachers:
        label, path = spec.split(":", 1)
        print(f"\n--- Loading {label}: {path}")
        dists, gts = load_teacher_labels(path)
        teachers_data[label] = (dists, gts)
        result = analyze_single_teacher(dists, gts, label)
        report["analyses"].append(result)

        print(f"  Samples: {result['n_samples']}")
        print(f"  Accuracy: {result['accuracy']}%")
        print(f"  FR distance to GT: {result['fr_to_gt_mean']:.4f} ± {result['fr_to_gt_std']:.4f}")
        print(f"  Entropy mean: {result['entropy_mean']:.4f} (ratio={result['entropy_ratio']:.4f})")
        print(f"  Peak prob mean: {result['peak_prob_mean']:.4f}")

    # Pairwise teacher distances
    if len(teachers_data) > 1:
        print(f"\n--- Pairwise Teacher Distances ---")
        pw = pairwise_teacher_distance(teachers_data)
        report["pairwise_distances"] = pw
        for pair, dist in pw.items():
            print(f"  {pair}: {dist:.4f}")

    # Fake vs real soft label comparison
    if args.fake_soft and args.real_soft:
        print(f"\n--- Fake vs Real Soft Label Comparison ---")
        fake_dists, fake_gts = load_teacher_labels(args.fake_soft)
        real_dists, real_gts = load_teacher_labels(args.real_soft)
        fvr = fake_vs_real_analysis(fake_dists, real_dists, fake_gts)
        report["fake_vs_real"] = fvr
        print(f"  Fake FR→GT mean: {fvr['fake_fr_to_gt_mean']:.4f}")
        print(f"  Real FR→GT mean: {fvr['real_fr_to_gt_mean']:.4f}")
        print(f"  Fake↔Real FR mean: {fvr['fake_real_fr_mean']:.4f}")
        print(f"  Fake entropy mean: {fvr['fake_entropy_mean']:.4f}")
        print(f"  Real entropy mean: {fvr['real_entropy_mean']:.4f}")
        print(f"  Real/Fake entropy ratio: {fvr['entropy_ratio_fake_real']:.4f}")

    # Save report
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[SAVED] {args.output}")

    # Summary table
    print("\n" + "=" * 70)
    print("Summary Table")
    print("=" * 70)
    print(f"{'Teacher':<20} {'Acc%':>6} {'FR→GT':>8} {'±std':>8} {'H_mean':>8} {'H_ratio':>8} {'Peak':>6}")
    print("-" * 70)
    for r in report["analyses"]:
        print(f"{r['label']:<20} {r['accuracy']:>6.2f} {r['fr_to_gt_mean']:>8.4f} "
              f"{r['fr_to_gt_std']:>8.4f} {r['entropy_mean']:>8.4f} {r['entropy_ratio']:>8.4f} "
              f"{r['peak_prob_mean']:>6.4f}")


if __name__ == "__main__":
    main()
