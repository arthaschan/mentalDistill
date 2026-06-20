#!/usr/bin/env python3
"""
E2: 跨数据集指标预测一致性分析 (便宜, 只需 teacher logprobs)

读取 teacher_labels_ext/<dataset>_<teacher>_logprobs.jsonl, 对每个数据集算各教师的
training-free 指标 (准确率/熵_auc/LEEP/LogME), 检验"教师排序是否跨数据集稳定"。

核心问题: 在 CMExam 上观察到的"指标排序 ~ 教师可靠性"是否在 MMLU/MedQA 上也成立?
若跨数据集一致 -> 规律普适 (上期刊关键证据)。

用法:
    python research/distillability/scripts/e2_cross_dataset.py
"""
import json
import os
import glob
import numpy as np

OPTION_LETTERS = ["A", "B", "C", "D", "E"]
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO = os.path.dirname(REPO)
DIST = os.path.join(REPO, "research", "distillability")
EXT = os.path.join(DIST, "teacher_labels_ext")
OUT = os.path.join(DIST, "outputs", "e2_cross_dataset.json")


def parse_dist(row):
    dist = row.get("TeacherDist", {})
    gt = row.get("OriginalAnswer") or row.get("Answer", "")
    gt = str(gt).strip().upper()
    if not dist or gt not in OPTION_LETTERS:
        return None, None
    raw = [float(dist.get(ch, 0.0)) for ch in OPTION_LETTERS]
    if sum(raw) <= 1e-9:
        return None, None
    p = np.array([max(v, 1e-12) for v in raw], dtype=np.float64)
    p = p / p.sum()
    return p, OPTION_LETTERS.index(gt)


def entropy_auc(thetas, correct):
    """用 -熵 (越确定越可能对) 预测对错的 AUC。无 sklearn 时手算 rank-AUC。"""
    ent = np.array([-np.sum(t * np.log(t + 1e-12)) for t in thetas])
    score = -ent  # 越大越predicted-correct
    y = np.array(correct, dtype=int)
    n_pos, n_neg = y.sum(), (1 - y).sum()
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    auc = (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def leep(theta, y):
    n, Kz = theta.shape
    Ky = int(y.max()) + 1
    P_zy = np.zeros((Kz, Ky))
    for c in range(Ky):
        m = (y == c)
        if m.any():
            P_zy[:, c] = theta[m].sum(axis=0)
    P_zy /= n
    P_z = P_zy.sum(axis=1, keepdims=True)
    P_y_z = P_zy / np.clip(P_z, 1e-12, None)
    cond = theta @ P_y_z
    px = cond[np.arange(n), y]
    return float(np.mean(np.log(np.clip(px, 1e-12, None))))


def logme(F, y, max_iter=50):
    F = (F - F.mean(axis=0, keepdims=True)).astype(np.float64)
    N, D = F.shape
    uu, ss, vh = np.linalg.svd(F.T @ F)
    s = np.sqrt(np.clip(ss, 0, None))
    k = int((s > 1e-10).sum())
    s = s[:k]; vh = vh[:k]
    u = (F @ vh.T)[:, :k] / s.reshape(1, -1)
    sigma = s ** 2
    Ky = int(y.max()) + 1
    ev = []
    for c in range(Ky):
        y_ = (y == c).astype(np.float64); y_ -= y_.mean()
        x = u.T @ y_; x2 = x ** 2
        res_x2 = float((y_ ** 2).sum() - x2.sum())
        alpha, beta = 1.0, 1.0; t = 1.0
        m2 = res2 = 0.0
        for _ in range(max_iter):
            gamma = float((sigma / (sigma + t)).sum())
            m2 = float((sigma * x2 / ((t + sigma) ** 2)).sum())
            res2 = float((x2 / ((1 + sigma / t) ** 2)).sum() + res_x2)
            alpha = gamma / (m2 + 1e-8); beta = (N - gamma) / (res2 + 1e-8)
            tn = alpha / beta
            if abs(tn - t) / max(t, 1e-12) <= 1e-4:
                t = tn; break
            t = tn
        e = (k / 2 * np.log(alpha) + N / 2 * np.log(beta)
             - 0.5 * np.sum(np.log(alpha + beta * sigma))
             - beta / 2 * res2 - alpha / 2 * m2 - N / 2 * np.log(2 * np.pi)) / N
        ev.append(float(e))
    return float(np.mean(ev))


def load(path):
    thetas, ys = [], []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        p, gt = parse_dist(row)
        if p is not None:
            thetas.append(p); ys.append(gt)
    return np.array(thetas), np.array(ys, dtype=int)


def main():
    files = sorted(glob.glob(os.path.join(EXT, "*_logprobs.jsonl")))
    if not files:
        print(f"[等待] {EXT} 下还没有 logprobs 文件 (编排器 E2 步骤未完成)。")
        return
    # 按数据集分组
    by_ds = {}
    for f in files:
        base = os.path.basename(f).replace("_logprobs.jsonl", "")
        # 格式 <dataset>_<teacher>, dataset 可能含下划线(mmlu_med)
        parts = base.rsplit("_", 1)
        if len(parts) != 2:
            continue
        ds, teacher = parts
        by_ds.setdefault(ds, {})[teacher] = f

    results = {}
    for ds, teachers in sorted(by_ds.items()):
        print("=" * 78)
        print(f"数据集: {ds}")
        print("=" * 78)
        print(f"{'teacher':10s}{'n':>7}{'acc%':>8}{'ent_auc':>9}{'LEEP':>9}{'LogME':>9}")
        results[ds] = {}
        for teacher, path in sorted(teachers.items()):
            theta, y = load(path)
            if len(theta) == 0:
                continue
            correct = (theta.argmax(axis=1) == y)
            acc = float(correct.mean() * 100)
            eauc = entropy_auc(theta, correct)
            lp = leep(theta, y)
            lm = logme(np.log(theta), y)
            results[ds][teacher] = {
                "n": int(len(theta)), "acc": round(acc, 2),
                "entropy_auc": round(eauc, 4) if eauc else None,
                "leep": round(lp, 4), "logme": round(lm, 4),
            }
            ea = f"{eauc:.4f}" if eauc else "  n/a"
            print(f"{teacher:10s}{len(theta):>7}{acc:>8.2f}{ea:>9}{lp:>9.4f}{lm:>9.4f}")
        print()

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(results, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[SAVED] {OUT}")
    print("\n判读: 比较各数据集内的教师排序是否一致 (尤其 acc 排序)。")
    print("      若跨数据集稳定 -> '教师可靠性预测可蒸馏性'是普适规律, 上期刊关键证据。")


if __name__ == "__main__":
    main()
