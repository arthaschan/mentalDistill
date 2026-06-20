#!/usr/bin/env python3
"""
迁移性指标打分器: LogME + LEEP (training-free 教师可蒸馏性预测的必备对标基线)

动机: 审稿人一定会问"为什么不用 LogME/LEEP?"。本脚本把迁移学习社区成熟的
training-free 迁移性指标搬到"蒸馏教师选择"场景, 与你的几何 DI / 教师准确率 / 熵
放进同一张指标家族对比表, 用同样的真实蒸馏增益做排序相关。

输入: 每个教师的 logprobs jsonl (含 TeacherDist 5维分布 + OriginalAnswer GT)。
  - LEEP : 源类别=教师预测的 A-E 软分布, 目标类别=GT 的 A-E。完全契合 (同标签空间)。
  - LogME: 以教师的 log-prob 向量(5维)作为特征 f(x), GT 作为标签, 算 log maximum evidence。
           5维特征略退化但仍是合法且常被引用的迁移性分数 (会标注此 caveat)。

参考实现:
  LogME: You et al., "LogME: Practical Assessment of Pre-trained Models for
         Transfer Learning", ICML 2021 (定点迭代版, 改写自官方 thuml/LogME)。
  LEEP : Nguyen et al., "LEEP: A New Measure to Evaluate Transferability of
         Learned Representations", ICML 2020。

用法:
    python research/distillability/scripts/transferability_scores.py            # 跑全部教师
    python research/distillability/scripts/transferability_scores.py --selftest # 仅算法自检
"""
import argparse
import json
import os
import numpy as np

OPTION_LETTERS = ["A", "B", "C", "D", "E"]
K = len(OPTION_LETTERS)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO = os.path.dirname(REPO)  # repo root
DIST = os.path.join(REPO, "research", "distillability")
LABELS_DIR = os.path.join(DIST, "teacher_labels")
OUT = os.path.join(DIST, "outputs", "transferability_scores.json")

# 教师 label -> logprobs 文件 (Llama70B pipeline 不一致, 仅作敏感性, 默认不进主表)
TEACHERS = {
    "Qwen32B":  os.path.join(LABELS_DIR, "qwen32b_train_logprobs.jsonl"),
    "Qwen14B":  os.path.join(LABELS_DIR, "qwen14b_train_logprobs.jsonl"),
    "GLM32B":   os.path.join(LABELS_DIR, "glm32b_train_logprobs.jsonl"),
    "Yi34B":    os.path.join(LABELS_DIR, "yi34b_train_logprobs.jsonl"),
    "Phi4":     os.path.join(LABELS_DIR, "phi4_train_logprobs.jsonl"),
    "Gemma27B": os.path.join(LABELS_DIR, "gemma27b_train_logprobs.jsonl"),
}


def parse_dist(row):
    """复用 teacher_distillability_score.py 的解析逻辑, 返回 (probs[5], gt_idx)."""
    dist = row.get("TeacherDist", {})
    gt = row.get("OriginalAnswer") or row.get("Answer", "")
    gt = str(gt).strip().upper()
    if not dist or gt not in OPTION_LETTERS:
        return None, None
    raw = [float(dist.get(ch, 0.0)) for ch in OPTION_LETTERS]
    if sum(raw) <= 1e-9:
        return None, None
    probs = np.array([max(v, 1e-12) for v in raw], dtype=np.float64)
    probs = probs / probs.sum()
    return probs, OPTION_LETTERS.index(gt)


def load_teacher(path):
    """返回 (theta[n,5] 教师软分布, y[n] GT类别索引)."""
    thetas, ys = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            p, gt = parse_dist(row)
            if p is None:
                continue
            thetas.append(p)
            ys.append(gt)
    return np.array(thetas), np.array(ys, dtype=int)


# --------------------------- LEEP ---------------------------
def leep(theta, y):
    """
    LEEP score (越大越可迁移).
    theta: n×Kz 源模型软预测; y: n 目标类别索引 (Ky 类).
    P(y,z) = mean_x theta_z(x) * 1[y_x=y]; P(z)=sum_y P(y,z); P(y|z)=P(y,z)/P(z)
    LEEP = mean_x log( sum_z P(y_x|z) theta_z(x) )
    """
    n, Kz = theta.shape
    Ky = int(y.max()) + 1
    # 联合 P(z, y)
    P_zy = np.zeros((Kz, Ky), dtype=np.float64)
    for c in range(Ky):
        mask = (y == c)
        if mask.any():
            P_zy[:, c] = theta[mask].sum(axis=0)
    P_zy /= n
    P_z = P_zy.sum(axis=1, keepdims=True)  # Kz×1
    P_y_given_z = P_zy / np.clip(P_z, 1e-12, None)  # Kz×Ky
    # 每个样本: sum_z P(y_x|z) theta_z(x)
    cond = theta @ P_y_given_z  # n×Ky, cond[x, c] = sum_z P(c|z) theta_z(x)
    px = cond[np.arange(n), y]  # 取真实标签那一列
    return float(np.mean(np.log(np.clip(px, 1e-12, None))))


# --------------------------- LogME ---------------------------
def _truncated_svd(x):
    """官方 thuml/LogME 的稳健 SVD: 经 x^T x 求特征, 数值更稳。返回 u(N×k), s(k), vh(k×D)."""
    uu, ss, vh = np.linalg.svd(x.T @ x)
    s = np.sqrt(np.clip(ss, 0, None))
    u_times_sigma = x @ vh.T
    k = int((s > 1e-10).sum())
    s = s[:k]
    vh = vh[:k]
    u = u_times_sigma[:, :k] / s.reshape(1, -1)
    return u, s, vh


def logme(F, y, max_iter=50):
    """
    LogME score (越大越可迁移). 改写自 You et al. 2021 官方定点迭代实现。
    F: n×D 特征; y: n 类别索引。多类做 one-vs-rest 取证据均值。
    """
    F = F.astype(np.float64)
    # 标准化特征 (官方做法)
    F = (F - F.mean(axis=0, keepdims=True))
    N, D = F.shape
    u, s, vh = _truncated_svd(F)
    sigma = (s ** 2)  # k
    k = sigma.shape[0]
    Ky = int(y.max()) + 1
    evidences = []
    for c in range(Ky):
        y_ = (y == c).astype(np.float64)
        y_ = y_ - y_.mean()  # 中心化 (与特征一致)
        x = u.T @ y_              # k
        x2 = x ** 2
        res_x2 = float((y_ ** 2).sum() - x2.sum())
        alpha, beta = 1.0, 1.0
        t = alpha / beta
        gamma, m2, res2 = 0.0, 0.0, res_x2
        for _ in range(max_iter):
            gamma = float((sigma / (sigma + t)).sum())
            m2 = float((sigma * x2 / ((t + sigma) ** 2)).sum())
            res2 = float((x2 / ((1 + sigma / t) ** 2)).sum() + res_x2)
            alpha = gamma / (m2 + 1e-8)
            beta = (N - gamma) / (res2 + 1e-8)
            t_new = alpha / beta
            if abs(t_new - t) / max(t, 1e-12) <= 1e-4:
                t = t_new
                break
            t = t_new
        evidence = (k / 2 * np.log(alpha)
                    + N / 2 * np.log(beta)
                    - 0.5 * np.sum(np.log(alpha + beta * sigma))
                    - beta / 2 * res2
                    - alpha / 2 * m2
                    - N / 2 * np.log(2 * np.pi)) / N
        evidences.append(float(evidence))
    return float(np.mean(evidences))


# --------------------------- 自检 ---------------------------
def selftest():
    """合成数据: 信息量大的特征应得到更高的 LogME/LEEP。"""
    rng = np.random.default_rng(0)
    n = 2000
    y = rng.integers(0, K, size=n)
    # 强信号: 教师分布高度集中在正确类 (acc~0.9)
    strong = np.full((n, K), 0.02)
    for i in range(n):
        correct = rng.random() < 0.9
        cls = y[i] if correct else rng.integers(0, K)
        strong[i, cls] = 0.92
    strong /= strong.sum(axis=1, keepdims=True)
    # 弱信号: 接近均匀 (acc~0.3)
    weak = np.full((n, K), 0.18)
    for i in range(n):
        if rng.random() < 0.3:
            weak[i, y[i]] = 0.28
    weak /= weak.sum(axis=1, keepdims=True)

    print("=== LogME/LEEP 算法自检 (强信号应 > 弱信号) ===")
    for name, theta in [("strong(acc~0.9)", strong), ("weak(acc~0.3)", weak)]:
        F = np.log(theta)  # log-prob 作为 LogME 特征
        print(f"  {name:18s}: LEEP={leep(theta, y):+.4f}  LogME={logme(F, y):+.4f}")
    print("  -> 若 strong 的两个分数都明显高于 weak, 算法实现正确。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="仅运行算法自检")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return

    results = {}
    print("=" * 78)
    print("Training-free 迁移性指标: LogME + LEEP (越大越可迁移/可蒸馏)")
    print("=" * 78)
    print(f"{'teacher':10s}{'n':>7}{'teacher_acc':>13}{'LEEP':>10}{'LogME':>10}")
    for label, path in TEACHERS.items():
        if not os.path.exists(path):
            print(f"{label:10s}  [缺文件: {path}]")
            continue
        theta, y = load_teacher(path)
        if len(theta) == 0:
            print(f"{label:10s}  [无有效样本]")
            continue
        acc = float((theta.argmax(axis=1) == y).mean() * 100)
        F = np.log(theta)
        lp = leep(theta, y)
        lm = logme(F, y)
        results[label] = {"n": int(len(theta)), "teacher_acc": round(acc, 2),
                          "leep": round(lp, 4), "logme": round(lm, 4)}
        print(f"{label:10s}{len(theta):>7}{acc:>13.2f}{lp:>10.4f}{lm:>10.4f}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(results, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n[SAVED] {OUT}")
    print("\n注: LEEP 在此场景天然契合(源/目标同为A-E标签空间); LogME用5维log-prob特征略退化,")
    print("    作为迁移性家族对标基线纳入, 最终与 DI/准确率/熵 一起做排序相关 (见 h1_baseline_comparison)。")


if __name__ == "__main__":
    main()
