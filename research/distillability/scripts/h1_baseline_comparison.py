#!/usr/bin/env python3
"""
H1 前瞻性验证 + 朴素基线对照 (方向B 核心分水岭实验)

回答唯一关键问题: 几何可蒸馏性指数(DI) 预测蒸馏收益的能力,
是否显著强于两个免费的朴素基线 —— 教师准确率、教师输出熵(entropy_auc)?

- 若 DI 明显赢 -> 几何有独立科研价值, 论文走"几何版"
- 若打平/输   -> 诚实结论"教师准确率已足够", 论文走更稳的"准确率基线版"

自动从 runs 目录解析每个教师 3-seed 的真实 TEST-BEST 准确率, 计算:
  gain_vs_baseline = mean(geom) - mean(baseline)   # 学生相对零样本基线的净收益
  geom_minus_random = mean(geom) - mean(random)    # 几何筛选相对随机的净收益
然后对 [DI, teacher_acc, entropy_auc] 各算 Spearman/Kendall 排序相关。

用法:
    python research/distillability/scripts/h1_baseline_comparison.py
"""
import json
import os
import re
import glob
import statistics as st
from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO = os.path.dirname(REPO)  # -> repo root
DIST = os.path.join(REPO, "research", "distillability")
RUNS = os.path.join(DIST, "runs")
OUT = os.path.join(DIST, "outputs")

# 教师先验指标 (logprobs-only, 已冻结). entropy_auc 取自 combined_predictor_<label>.json 的单特征 entropy AUC.
PRIOR = {
    # label: (teacher_acc, geom_auc, distillability_index, entropy_auc)
    "Qwen32B":  (89.43, 0.8694, 3.9210, 0.876),
    "Qwen14B":  (86.00, 0.8563, 2.6560, None),   # entropy_auc 缺, 单算时跳过
    "GLM32B":   (83.29, 0.8785, 2.3125, 0.880),
    "Yi34B":    (77.39, 0.8150, -0.3076, 0.821),
    "Gemma27B": (57.60, 0.7614, -3.3477, 0.758),
    "Phi4":     (54.51, 0.7453, -4.3992, 0.747),
    # Llama70B 因 pipeline 不一致, 单独作敏感性, 默认不进主分析
}

TEST_RE = re.compile(r"test_acc=([0-9.]+)%")

# 迁移性指标 (LEEP/LogME) 从 transferability_scores.py 的输出动态加载, 保持同步。
TRANSFER_JSON = os.path.join(OUT, "transferability_scores.json")


def load_transfer():
    """返回 {label: {'leep':.., 'logme':..}}, 文件不存在则空 dict。"""
    if not os.path.exists(TRANSFER_JSON):
        return {}
    try:
        return json.load(open(TRANSFER_JSON, encoding="utf-8"))
    except Exception:
        return {}


def parse_test_acc(logpath):
    """返回该 log 的最后一个 TEST-BEST test_acc, 没有则 None."""
    if not os.path.exists(logpath):
        return None
    best = None
    with open(logpath, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "[TEST-BEST]" in line:
                m = TEST_RE.search(line)
                if m:
                    best = float(m.group(1))
    return best


def collect_teacher_gains():
    """
    扫描 runs/, 聚合每个教师 3-seed 的 baseline/geom/random 三臂均值.
    支持两种目录结构:
      - phase2_<label>/logs/<arm>_seed<seed>.log   (Yi/Gemma 用 run_phase2_distill.sh)
      - <ts>_<label>/logs/<arm>.log                (GLM/Phi 早期单seed-per-run)
    返回 {label: {'baseline':[..], 'geom':[..], 'random':[..]}}
    """
    arms = {"baseline_all": "baseline", "geom_top50": "geom", "random_top50": "random"}
    data = {}

    # 结构1: phase2_<label>
    for d in glob.glob(os.path.join(RUNS, "phase2_*")):
        label = os.path.basename(d).replace("phase2_", "")
        logdir = os.path.join(d, "logs")
        if not os.path.isdir(logdir):
            continue
        bucket = data.setdefault(label, {"baseline": [], "geom": [], "random": []})
        for armfile, key in arms.items():
            for lp in glob.glob(os.path.join(logdir, f"{armfile}_seed*.log")):
                acc = parse_test_acc(lp)
                if acc is not None:
                    bucket[key].append(acc)

    # 结构2: <ts>_<label> (每个 run 一个 seed, arm 日志名无 seed 后缀)
    for d in glob.glob(os.path.join(RUNS, "20*_*")):
        base = os.path.basename(d)
        label = base.split("_", 2)[-1]  # 20260619_195451_glm32b -> glm32b
        logdir = os.path.join(d, "logs")
        if not os.path.isdir(logdir):
            continue
        bucket = data.setdefault(label, {"baseline": [], "geom": [], "random": []})
        for armfile, key in arms.items():
            acc = parse_test_acc(os.path.join(logdir, f"{armfile}.log"))
            if acc is not None:
                bucket[key].append(acc)

    return data


# label 归一: runs 目录用小写, PRIOR 用驼峰
LABEL_MAP = {
    "yi34b": "Yi34B", "gemma27b": "Gemma27B", "glm32b": "GLM32B",
    "phi4": "Phi4", "qwen32b": "Qwen32B", "qwen14b": "Qwen14B",
}


def main():
    raw = collect_teacher_gains()
    transfer = load_transfer()
    rows = []  # (label, t_acc, geom_auc, DI, entropy_auc, leep, logme, gain_vs_base, geom_minus_random, n_seeds)
    print("=" * 92)
    print("每个教师的真实蒸馏结果 (3-seed 均值)")
    print("=" * 92)
    print(f"{'teacher':10s}{'n_seed':>7}{'baseline':>10}{'geom':>9}{'random':>9}"
          f"{'gain_vs_base':>14}{'geom-random':>13}")
    for lbl_lc, b in sorted(raw.items()):
        label = LABEL_MAP.get(lbl_lc, lbl_lc)
        if label not in PRIOR:
            continue
        if not (b["baseline"] and b["geom"] and b["random"]):
            print(f"{label:10s}  [未完成: baseline={len(b['baseline'])} "
                  f"geom={len(b['geom'])} random={len(b['random'])}]")
            continue
        mb, mg, mr = st.mean(b["baseline"]), st.mean(b["geom"]), st.mean(b["random"])
        gvb = mg - mb
        gmr = mg - mr
        n = min(len(b["baseline"]), len(b["geom"]), len(b["random"]))
        tacc, gauc, di, eauc = PRIOR[label]
        tr = transfer.get(label, {})
        leep_v = tr.get("leep")
        logme_v = tr.get("logme")
        rows.append((label, tacc, gauc, di, eauc, leep_v, logme_v, gvb, gmr, n))
        print(f"{label:10s}{n:>7}{mb:>10.2f}{mg:>9.2f}{mr:>9.2f}{gvb:>+14.2f}{gmr:>+13.2f}")

    if len(rows) < 3:
        print(f"\n[等待] 已完成教师数={len(rows)} (<3), 蒸馏跑完后再运行本脚本出相关性。")
        return

    print("\n" + "=" * 92)
    print(f"H1 排序相关: 三个预测器 vs 真实蒸馏收益  (N={len(rows)} 教师)")
    print("=" * 92)

    def corr_block(pred_name, pred_vals, target_name, target_vals):
        # 过滤 None (如 Qwen14B 无 entropy_auc)
        pairs = [(p, t) for p, t in zip(pred_vals, target_vals) if p is not None]
        if len(pairs) < 3:
            return f"  {pred_name:14s} -> {target_name:18s}: N<3, 跳过"
        ps, ts = zip(*pairs)
        sp = stats.spearmanr(ps, ts)
        kt = stats.kendalltau(ps, ts)
        return (f"  {pred_name:14s} -> {target_name:18s} (N={len(pairs)}): "
                f"Spearman ρ={sp.correlation:+.3f} (p={sp.pvalue:.3f}) | "
                f"Kendall τ={kt.correlation:+.3f} (p={kt.pvalue:.3f})")

    labels = [r[0] for r in rows]
    t_acc = [r[1] for r in rows]
    di = [r[3] for r in rows]
    e_auc = [r[4] for r in rows]
    leep_v = [r[5] for r in rows]
    logme_v = [r[6] for r in rows]
    gain_base = [r[7] for r in rows]
    geom_rand = [r[8] for r in rows]

    print(f"教师集: {labels}\n")
    for target_name, target in [("gain_vs_baseline", gain_base),
                                ("geom_minus_random", geom_rand)]:
        print(f"[目标: {target_name}]")
        print(corr_block("DI(几何)", di, target_name, target))
        print(corr_block("teacher_acc", t_acc, target_name, target))
        print(corr_block("entropy_auc", e_auc, target_name, target))
        print(corr_block("LEEP", leep_v, target_name, target))
        print(corr_block("LogME", logme_v, target_name, target))
        print()

    print("=" * 92)
    print("判读 (指标家族对比):")
    print("  - 比较 5 个 training-free 指标 (DI几何/准确率/熵/LEEP/LogME) 谁的 |ρ| 最高")
    print("  - 若 DI 明显胜出 -> 几何有独立价值, 走'几何版'论文")
    print("  - 若它们接近 -> 诚实结论'可蒸馏性主要由教师整体可靠性单变量驱动', benchmark 式贡献")
    print("  - 务必对标 LEEP/LogML: 这是迁移性社区的标准基线, 防审稿人'为何不用 LogME'")
    print("  - N 很小, 务必报告 p 值与置信区间, 不要只看点估计")
    print("=" * 92)

    # 落盘
    out = {
        "n_teachers": len(rows),
        "teachers": labels,
        "rows": [
            {"label": r[0], "teacher_acc": r[1], "geom_auc": r[2], "DI": r[3],
             "entropy_auc": r[4], "leep": r[5], "logme": r[6],
             "gain_vs_baseline": round(r[7], 3),
             "geom_minus_random": round(r[8], 3), "n_seeds": r[9]}
            for r in rows
        ],
    }
    os.makedirs(OUT, exist_ok=True)
    outpath = os.path.join(OUT, "h1_baseline_comparison.json")
    json.dump(out, open(outpath, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n[SAVED] {outpath}")


if __name__ == "__main__":
    main()
