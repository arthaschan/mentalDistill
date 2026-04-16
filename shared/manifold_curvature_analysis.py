#!/usr/bin/env python3
"""
manifold_curvature_analysis.py — 概率单纯形上教师标签的流形曲率分析

在 5-选项概率单纯形 Δ⁴ 上，Fisher 信息度量赋予其球面几何（与 S⁴(2) 的一部分等距）。
本脚本计算每个教师标签的局部几何特征：

1. Fisher 度量张量行列式 det(g) = 1/∏pᵢ → 衡量局部体积膨胀
2. 黎曼体积元素 √det(g) → 该点处的体积密度
3. 有效维度：标签在单纯形内部的分散程度
4. 边界距离：标签到单纯形边界的测地线距离
5. 局部曲率效应：正曲率(K=1/4)对测地球体积的修正

理论背景：
- 5-选项多项式模型的 Fisher 度量：g_ij(p) = δ_ij / p_i
- 截面曲率 K = 1/4（常数正曲率）
- Ricci 曲率 Ric = (n-2)/4 · g，其中 n=5
- 标量曲率 R = n(n-1)(n-2) / (4(n-1)) = (n-2)n/4 = 15/4 = 3.75
  实际上对 Δ^(n-1) with n categories: R = (n-1)(n-2)/4
  n=5 → R = 4*3/4 = 3
"""
import argparse
import json
import math
import os

OPTION_LETTERS = ["A", "B", "C", "D", "E"]
N_OPTIONS = len(OPTION_LETTERS)

# 概率单纯形 Δ^(n-1) 的常数截面曲率
SECTIONAL_CURVATURE = 0.25  # K = 1/4
# 等距球面 S^(n-1)(r) 的半径 r=2
SPHERE_RADIUS = 2.0
# 单纯形维度
DIM = N_OPTIONS - 1  # = 4


def parse_dist(row):
    """从 JSONL 记录中提取教师分布和 GT。"""
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


def load_teacher_labels(path):
    """加载 teacher label JSONL 文件。"""
    dists, gts = [], []
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


# === 局部度量张量分析 ===

def fisher_metric_determinant(p):
    """Fisher 度量张量行列式 det(g) = 1 / ∏ p_i
    
    在多项式模型中，Fisher 度量是对角阵 g_ij = δ_ij / p_i，
    因此 det(g) = ∏ (1/p_i) = 1 / ∏ p_i。
    
    物理含义：det(g) 衡量局部体积膨胀因子。
    - 靠近单纯形边界（某个 p_i → 0）时 det(g) → ∞，体积极度膨胀
    - 在单纯形中心（均匀分布 p_i = 1/n）时 det(g) = n^n 最小
    """
    prod = 1.0
    for pi in p:
        prod *= pi
    return 1.0 / max(prod, 1e-60)


def fisher_volume_element(p):
    """黎曼体积元素 √det(g) = 1 / √(∏ p_i)
    
    体积密度：在 Fisher 度量下，该点周围单位坐标体积对应多大的几何体积。
    """
    return math.sqrt(fisher_metric_determinant(p))


def log_fisher_volume_element(p):
    """对数体积元素 log₁₀(√det(g))，方便比较数量级。"""
    return math.log10(fisher_volume_element(p))


# === 边界距离分析 ===

def min_prob(p):
    """最小概率值——衡量离单纯形边界的"坐标距离"。"""
    return min(p)


def boundary_geodesic_distance(p):
    """到最近边界面的 Fisher-Rao 测地线距离的近似。
    
    单纯形的边界面对应某个 p_i = 0。
    从 p 到 p_i=0 面的 FR 距离为 2·arcsin(√p_min)（对最小分量）。
    
    直觉：坐标上很小的概率差（如 p_i=0.01 vs 0），在 Fisher 度量下
    对应很大的测地线距离，因为 g_ii = 1/p_i 在 p_i→0 时趋于无穷。
    """
    p_min = min(p)
    return 2.0 * math.asin(math.sqrt(max(p_min, 1e-12)))


def distance_to_center(p):
    """到均匀分布（单纯形中心）的 Fisher-Rao 距离。
    
    d_FR(p, uniform) = 2·arccos(Σ √(p_i · 1/n))
                     = 2·arccos(Σ √p_i / √n)
    """
    n = len(p)
    bc = sum(math.sqrt(pi / n) for pi in p)
    bc = min(bc, 1.0)
    return 2.0 * math.acos(bc)


# === 曲率效应分析 ===

def geodesic_ball_volume_ratio(radius):
    """正曲率空间中测地球体积与平坦空间的比值。
    
    在常曲率 K 的 d 维空间中，半径 r 的测地球体积为：
    V_K(r) / V_0(r) ≈ 1 - K·d·r² / (6(d+2)) + O(r⁴)
    
    K > 0（正曲率）→ 比值 < 1，测地球比平坦空间的球更小。
    直觉：正曲率"压缩"了远处的空间，就像地球表面上的圆面积小于平面上的。
    
    对我们的单纯形：K=1/4, d=4
    """
    K = SECTIONAL_CURVATURE
    d = DIM
    correction = 1.0 - K * d * radius**2 / (6.0 * (d + 2))
    return max(correction, 0.01)  # 大半径时近似失效，clamp


def effective_dimension(dists):
    """通过 PCA 估计标签在单纯形中的有效维度。
    
    将概率分布映射到球面坐标 ξ_i = 2√p_i（Fisher 度量的等距嵌入），
    然后计算协方差矩阵的有效维度 = (Σλ_i)² / Σλ_i²。
    """
    # 球面嵌入
    embedded = []
    for p in dists:
        xi = [2.0 * math.sqrt(pi) for pi in p]
        embedded.append(xi)
    
    n = len(embedded)
    d = len(embedded[0])
    
    # 计算均值
    mean = [sum(embedded[i][j] for i in range(n)) / n for j in range(d)]
    
    # 计算协方差矩阵
    cov = [[0.0] * d for _ in range(d)]
    for i in range(n):
        centered = [embedded[i][j] - mean[j] for j in range(d)]
        for a in range(d):
            for b in range(d):
                cov[a][b] += centered[a] * centered[b]
    for a in range(d):
        for b in range(d):
            cov[a][b] /= n
    
    # 计算特征值（幂法近似前几个主成分即可）
    # 用简单的迹和 Frobenius 范数估计有效维度
    trace = sum(cov[i][i] for i in range(d))
    frobenius_sq = sum(cov[i][j]**2 for i in range(d) for j in range(d))
    
    if frobenius_sq < 1e-15:
        return 0.0
    
    # 有效维度 = trace² / frobenius²
    eff_dim = trace**2 / frobenius_sq
    return eff_dim


def concentration_on_simplex(dists):
    """计算标签在概率单纯形上的集中度。
    
    使用 Fisher-Rao 距离的方差来衡量分散程度。
    高集中度（低方差）= 标签聚集在单纯形的一小块区域。
    """
    n = len(dists)
    # 计算 Fréchet 均值的近似（欧氏均值投影到单纯形）
    mean_p = [sum(dists[i][j] for i in range(n)) / n for j in range(N_OPTIONS)]
    
    # 计算到均值的 FR 距离
    fr_to_mean = []
    for p in dists:
        bc = sum(math.sqrt(p[j] * mean_p[j]) for j in range(N_OPTIONS))
        bc = min(bc, 1.0)
        d = 2.0 * math.acos(bc)
        fr_to_mean.append(d)
    
    mean_d = sum(fr_to_mean) / n
    var_d = sum((d - mean_d)**2 for d in fr_to_mean) / n
    return mean_d, math.sqrt(var_d)


def analyze_teacher_curvature(dists, gts, label):
    """对单个教师的标签进行完整的流形曲率分析。"""
    n = len(dists)
    
    # 1. Fisher 度量张量分析
    log_det_vals = [log_fisher_volume_element(p) for p in dists]
    vol_elem_vals = [fisher_volume_element(p) for p in dists]
    
    # 2. 边界距离分析
    boundary_dists = [boundary_geodesic_distance(p) for p in dists]
    center_dists = [distance_to_center(p) for p in dists]
    min_probs = [min_prob(p) for p in dists]
    
    # 3. 有效维度
    eff_dim = effective_dimension(dists)
    
    # 4. 集中度
    mean_spread, std_spread = concentration_on_simplex(dists)
    
    # 5. 正确/错误样本的曲率差异
    correct_log_det = []
    wrong_log_det = []
    correct_boundary = []
    wrong_boundary = []
    for i, (p, gt) in enumerate(zip(dists, gts)):
        pred = OPTION_LETTERS[p.index(max(p))]
        if pred == gt:
            correct_log_det.append(log_det_vals[i])
            correct_boundary.append(boundary_dists[i])
        else:
            wrong_log_det.append(log_det_vals[i])
            wrong_boundary.append(boundary_dists[i])
    
    # 6. 曲率对测地球的影响
    mean_boundary = sum(boundary_dists) / n
    vol_ratio = geodesic_ball_volume_ratio(mean_boundary)
    
    result = {
        "label": label,
        "n_samples": n,
        # Fisher 度量
        "log10_vol_element_mean": round(sum(log_det_vals) / n, 4),
        "log10_vol_element_std": round(
            math.sqrt(sum((x - sum(log_det_vals)/n)**2 for x in log_det_vals) / n), 4
        ),
        "log10_vol_element_max": round(max(log_det_vals), 4),
        "log10_vol_element_min": round(min(log_det_vals), 4),
        # 边界距离
        "boundary_dist_mean": round(sum(boundary_dists) / n, 4),
        "boundary_dist_std": round(
            math.sqrt(sum((x - sum(boundary_dists)/n)**2 for x in boundary_dists) / n), 4
        ),
        "min_prob_mean": round(sum(min_probs) / n, 6),
        "min_prob_std": round(
            math.sqrt(sum((x - sum(min_probs)/n)**2 for x in min_probs) / n), 6
        ),
        # 到中心距离
        "center_dist_mean": round(sum(center_dists) / n, 4),
        "center_dist_std": round(
            math.sqrt(sum((x - sum(center_dists)/n)**2 for x in center_dists) / n), 4
        ),
        # 有效维度与集中度
        "effective_dimension": round(eff_dim, 4),
        "spread_mean": round(mean_spread, 4),
        "spread_std": round(std_spread, 4),
        # 曲率效应
        "geodesic_ball_vol_ratio": round(vol_ratio, 4),
        # 正确 vs 错误样本
        "correct_log_vol_mean": round(sum(correct_log_det) / len(correct_log_det), 4) if correct_log_det else None,
        "wrong_log_vol_mean": round(sum(wrong_log_det) / len(wrong_log_det), 4) if wrong_log_det else None,
        "correct_boundary_mean": round(sum(correct_boundary) / len(correct_boundary), 4) if correct_boundary else None,
        "wrong_boundary_mean": round(sum(wrong_boundary) / len(wrong_boundary), 4) if wrong_boundary else None,
        "n_correct": len(correct_log_det),
        "n_wrong": len(wrong_log_det),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Manifold curvature analysis of teacher labels on Δ⁴")
    parser.add_argument("--teachers", nargs="+", required=True,
                        help="label:path pairs, e.g. 'DeepSeek:02/data/train_head_distill.jsonl'")
    parser.add_argument("--output", type=str, default="outputs/curvature_report.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)

    report = {"manifold_properties": {}, "teachers": []}

    # 流形全局属性
    report["manifold_properties"] = {
        "space": f"Δ^{DIM} (probability simplex with {N_OPTIONS} categories)",
        "dimension": DIM,
        "metric": "Fisher information metric: g_ij(p) = δ_ij / p_i",
        "sectional_curvature": SECTIONAL_CURVATURE,
        "scalar_curvature": (N_OPTIONS - 1) * (N_OPTIONS - 2) / 4.0,
        "isometric_to": f"portion of S^{DIM}(2) (4-sphere of radius 2)",
        "center_point": f"uniform distribution (1/{N_OPTIONS}, ..., 1/{N_OPTIONS})",
        "center_det_g": N_OPTIONS ** N_OPTIONS,
        "center_log10_vol": round(math.log10(math.sqrt(N_OPTIONS ** N_OPTIONS)), 4),
    }

    print("=" * 80)
    print("Manifold Curvature Analysis on Δ⁴ (Fisher Geometry)")
    print("=" * 80)
    print(f"\nGlobal properties:")
    for k, v in report["manifold_properties"].items():
        print(f"  {k}: {v}")

    for spec in args.teachers:
        label, path = spec.split(":", 1)
        print(f"\n{'─'*80}")
        print(f"Teacher: {label} ({path})")
        print(f"{'─'*80}")
        dists, gts = load_teacher_labels(path)
        result = analyze_teacher_curvature(dists, gts, label)
        report["teachers"].append(result)

        print(f"  Samples: {result['n_samples']}")
        print(f"  log₁₀(√det g):  {result['log10_vol_element_mean']:.4f} ± {result['log10_vol_element_std']:.4f}  "
              f"[{result['log10_vol_element_min']:.4f}, {result['log10_vol_element_max']:.4f}]")
        print(f"  Boundary dist:   {result['boundary_dist_mean']:.4f} ± {result['boundary_dist_std']:.4f}")
        print(f"  Min prob:        {result['min_prob_mean']:.6f} ± {result['min_prob_std']:.6f}")
        print(f"  Center dist:     {result['center_dist_mean']:.4f} ± {result['center_dist_std']:.4f}")
        print(f"  Eff. dimension:  {result['effective_dimension']:.4f}")
        print(f"  FR spread:       {result['spread_mean']:.4f} ± {result['spread_std']:.4f}")
        print(f"  Geod ball ratio: {result['geodesic_ball_vol_ratio']:.4f}")
        if result['correct_log_vol_mean'] is not None and result['wrong_log_vol_mean'] is not None:
            print(f"  Correct samples ({result['n_correct']}): log_vol={result['correct_log_vol_mean']:.4f}, "
                  f"boundary={result['correct_boundary_mean']:.4f}")
            print(f"  Wrong samples   ({result['n_wrong']}):  log_vol={result['wrong_log_vol_mean']:.4f}, "
                  f"boundary={result['wrong_boundary_mean']:.4f}")

    # Summary table
    print(f"\n{'='*80}")
    print("Summary Comparison")
    print(f"{'='*80}")
    print(f"{'Teacher':<18} {'log₁₀√detg':>12} {'BoundDist':>10} {'CenterDist':>11} {'EffDim':>7} {'Spread':>8} {'VolRatio':>9}")
    print("─" * 80)
    for r in report["teachers"]:
        print(f"{r['label']:<18} {r['log10_vol_element_mean']:>12.4f} {r['boundary_dist_mean']:>10.4f} "
              f"{r['center_dist_mean']:>11.4f} {r['effective_dimension']:>7.4f} {r['spread_mean']:>8.4f} "
              f"{r['geodesic_ball_vol_ratio']:>9.4f}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[SAVED] {args.output}")


if __name__ == "__main__":
    main()
