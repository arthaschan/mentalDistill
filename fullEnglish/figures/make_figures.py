#!/usr/bin/env python3
"""fullEnglish 结果图表生成 (IEEE 灰度印刷安全风格, 复用 thesis 风格).

数据来源 (结果文件, 随实验自动更新):
  - fig1 教师先验          -> 01_teacher_screening/reports/screening.json
  - fig2 熵=难度 moat      -> 02_fusion_oracle/entropy_difficulty.json
  - fig3 融合 oracle       -> 02_fusion_oracle/fusion_oracle.json
  - fig4 学生 vs 教师      -> 03_main_distill/runs/eval_results.json (实验完成后才生成)

运行: python3 fullEnglish/figures/make_figures.py
输出: fullEnglish/figures/*.png
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
FE = ROOT.parent
OUT_DIR = ROOT
OUT_DIR.mkdir(exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.dpi"] = 200
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 11
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 9
plt.rcParams["axes.edgecolor"] = "#222222"

FG_DARK = "#222222"
FG_MID = "#666666"
FG_LIGHT = "#b0b0b0"


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=FG_DARK)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT_DIR / name, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {OUT_DIR / name}")


def fig1_teacher_prior():
    d = load_json(FE / "01_teacher_screening/reports/screening.json")
    if not d:
        print("[skip fig1] 无 screening.json (先跑教师预评估)")
        return
    prior = d["teacher_prior"]
    order = d.get("order") or sorted(prior, key=lambda k: -prior[k]["acc"])
    student = d.get("student_name", "Qwen32B")
    names = list(reversed(order))  # 横向 bar 从下到上
    accs = [prior[n]["acc"] for n in names]
    colors = [FG_MID if n == student else (FG_DARK if n == order[0] else FG_LIGHT) for n in names]

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.barh(names, accs, color=colors, edgecolor=FG_DARK)
    for y, a in enumerate(accs):
        ax.text(a + 0.4, y, f"{a:.1f}%", va="center", fontsize=9)
    ax.set_xlabel("Zero-shot accuracy (%)")
    ax.set_title("Teacher screening on English medical MCQ (600 items)")
    ax.set_xlim(0, 100)
    style_axis(ax)
    save(fig, "fig1_teacher_prior.png")


def fig2_entropy_difficulty():
    d = load_json(FE / "02_fusion_oracle/entropy_difficulty.json")
    if not d:
        print("[skip fig2] 无 entropy_difficulty.json")
        return
    h4 = d["H4_entropy_locates_errors"]
    rho = d["5d_entropy_vs_consensus"]["mean_entropy_vs_consensus_rho"]
    grad = d["consensus_gradient"]

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))

    # 左: H4 err ratio
    names = list(h4.keys())
    ratios = [h4[n]["err_ratio"] for n in names]
    axes[0].bar(names, ratios, color=FG_MID, edgecolor=FG_DARK, hatch="//")
    axes[0].axhline(1.0, color=FG_DARK, linestyle="--", linewidth=1)
    axes[0].set_ylabel("Error rate ratio (high/low entropy)")
    axes[0].set_title("H4: entropy locates errors")
    axes[0].tick_params(axis="x", rotation=20)
    style_axis(axes[0])

    # 右: 共识梯度 (熵随错题数上升)
    ws = sorted(int(k) for k in grad.keys())
    ents = [grad[str(w)]["mean_teacher_ent"] for w in ws]
    ns = [grad[str(w)]["n"] for w in ws]
    axes[1].plot(ws, ents, "o-", color=FG_DARK, linewidth=1.5, markersize=6)
    for w, e, n in zip(ws, ents, ns):
        axes[1].annotate(f"n={n}", (w, e), textcoords="offset points", xytext=(0, 8), fontsize=7, ha="center")
    axes[1].set_xlabel("# teachers wrong (consensus difficulty)")
    axes[1].set_ylabel("Mean teacher entropy")
    axes[1].set_title(f"5d: entropy vs consensus (rho={rho})")
    style_axis(axes[1])
    save(fig, "fig2_entropy_difficulty.png")


def fig3_fusion_oracle():
    d = load_json(FE / "02_fusion_oracle/fusion_oracle.json")
    if not d:
        print("[skip fig3] 无 fusion_oracle.json")
        return
    best = d["best_single"]
    bs = d["best_single_acc"]
    fusion = d["fusion"]
    labels = ["best_single", "majority_vote", "conf_route", "prob_avg", "domain_route_CV", "oracle_anyright"]
    labels = [k for k in labels if k in fusion]
    vals = [fusion[k] for k in labels]
    colors = [FG_DARK if k == "best_single" else FG_LIGHT for k in labels]

    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    bars = ax.bar(labels, vals, color=colors, edgecolor=FG_DARK)
    ax.axhline(bs, color=FG_MID, linestyle="--", linewidth=1)
    ax.text(len(labels) - 0.5, bs + 0.4, f"best single = {bs:.1f}%", color=FG_MID, fontsize=8, ha="right")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v:.1f}", ha="center", fontsize=8)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Multi-teacher fusion ceiling (NO-GO)")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=20)
    style_axis(ax)
    save(fig, "fig3_fusion_oracle.png")


def fig4_student_vs_teacher():
    d = load_json(FE / "03_main_distill/runs/eval_results.json")
    if not d:
        print("[skip fig4] 无 eval_results.json (实验完成后自动生成)")
        return
    zs = d.get("zeroshot", {})
    adapters = d.get("adapters", {})
    # 取 a00 (DeepSeek 主线 α=0) 的均值
    a00 = [v["acc"] for k, v in adapters.items() if "a00" in k and "llama70b" not in k]
    sets = ["test_medqa", "test_medmcqa", "test_mmlu", "test_pubmedqa"]
    if not a00:
        print("[skip fig4] 无 a00 adapter")
        return
    mean = {s: float(np.mean([a[s] for a in a00 if s in a])) for s in sets}
    # 教师同集 (从标签文件)
    teach = {}
    for s in ["test_medqa", "test_medmcqa", "test_mmlu"]:
        p = FE / "03_main_distill/labels" / f"teacher_{s}.jsonl"
        if os.path.exists(p):
            n = c = 0
            for line in open(p):
                r = json.loads(line)
                gt = str(r.get("OriginalAnswer") or r.get("Answer", "")).strip().upper()
                ta = str(r.get("TeacherAnswer") or r.get("Answer", "")).strip().upper()
                if gt in "ABCDE" and ta in "ABCDE":
                    n += 1
                    c += int(ta == gt)
            teach[s] = round(100 * c / n, 2) if n else None
    x = np.arange(len(sets))
    t_vals = [teach.get(s) if s in teach else None for s in sets]
    s_vals = [mean.get(s) for s in sets]
    width = 0.35
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(x - width / 2, [v if v is not None else 0 for v in t_vals], width,
           label="Teacher", color=FG_LIGHT, edgecolor=FG_DARK, hatch="//")
    ax.bar(x + width / 2, [v if v is not None else 0 for v in s_vals], width,
           label="Student (alpha=0)", color=FG_MID, edgecolor=FG_DARK)
    for i, s in enumerate(sets):
        ax.text(i - width / 2, (t_vals[i] or 0) + 1, f"{t_vals[i]}" if t_vals[i] else "N/A", ha="center", fontsize=8)
        ax.text(i + width / 2, (s_vals[i] or 0) + 1, f"{s_vals[i]:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(sets)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Student vs teacher (same-set)")
    ax.set_ylim(0, 100)
    ax.legend()
    style_axis(ax)
    save(fig, "fig4_student_vs_teacher.png")


if __name__ == "__main__":
    print("生成 fullEnglish 图表:")
    fig1_teacher_prior()
    fig2_entropy_difficulty()
    fig3_fusion_oracle()
    fig4_student_vs_teacher()
    print("完成")
