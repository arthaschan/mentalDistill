#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "figures"
OUT_DIR.mkdir(exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.dpi"] = 200
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 11
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 9
plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["axes.edgecolor"] = "#222222"
plt.rcParams["grid.color"] = "#d0d0d0"
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.linewidth"] = 0.6

FG_DARK = "#222222"
FG_MID = "#666666"
FG_LIGHT = "#b0b0b0"


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=FG_DARK)
    ax.xaxis.label.set_color(FG_DARK)
    ax.yaxis.label.set_color(FG_DARK)
    ax.title.set_color(FG_DARK)


def save(fig, name: str):
    fig.tight_layout()
    fig.savefig(OUT_DIR / name, bbox_inches="tight")
    plt.close(fig)


def plot_teacher_quality_vs_gain():
    labels = ["Kimi", "Doubao", "DeepSeek", "Qwen-14B"]
    disagreement = np.array([27.83, 0.33, 14.14, 20.24])
    gain = np.array([0.0, 3.61, 4.82, -1.21])

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.scatter(disagreement, gain, s=90, color=FG_DARK, edgecolor="white", linewidth=0.8)

    for x, y, label in zip(disagreement, gain, labels):
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(6, 6))

    ax.axhline(0, color=FG_MID, linewidth=1, linestyle="--")
    ax.set_xlabel("Teacher-GT disagreement rate (%)")
    ax.set_ylabel("Student gain vs baseline (pp)")
    ax.set_title("Teacher quality shows a non-monotonic relation to distillation gain")
    style_axis(ax)
    save(fig, "fig_4_1_teacher_quality_vs_gain.png")


def plot_single_teacher_results():
    teachers = ["DeepSeek", "Doubao", "Kimi", "Qwen-14B", "Llama-70B*"]
    teacher_acc = np.array([87.95, 97.59, 62.65, 77.11, 72.45])
    student_best = np.array([81.93, 80.72, 77.11, 75.90, 87.59])
    student_mean = np.array([79.52, 79.52, 77.11, 75.50, 87.25])

    x = np.arange(len(teachers))
    width = 0.24

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.bar(x - width, teacher_acc, width, label="Teacher acc", color=FG_LIGHT, edgecolor=FG_DARK, hatch="//")
    ax.bar(x, student_best, width, label="Student best", color=FG_MID, edgecolor=FG_DARK)
    ax.bar(x + width, student_mean, width, label="Student stable", color="white", edgecolor=FG_DARK, hatch="..")

    ax.set_xticks(x)
    ax.set_xticklabels(teachers)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Single-teacher Choice-Head results")
    ax.legend(ncol=3, loc="upper center")
    ax.set_ylim(55, 102)
    style_axis(ax)

    ax.text(x[-1], 56.5, "* 991-question setting", ha="center", va="bottom", fontsize=8)
    save(fig, "fig_4_2_single_teacher_results.png")


def plot_full_vs_dental_baseline():
    models = ["Qwen-7B", "Qwen-14B", "DeepSeek-V3", "Llama-70B"]
    full_scores = np.array([76.49, 83.55, 87.18, 72.45])
    dental_scores = np.array([68.80, 74.40, 79.20, np.nan])

    x = np.arange(len(models))
    width = 0.32

    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    ax.bar(x - width / 2, full_scores, width, label="Full test set (991)", color=FG_MID, edgecolor=FG_DARK)

    valid = ~np.isnan(dental_scores)
    ax.bar(x[valid] + width / 2, dental_scores[valid], width, label="Dental subset (125)", color="white", edgecolor=FG_DARK, hatch="..")

    for idx, score in enumerate(dental_scores):
        if np.isnan(score):
            ax.text(x[idx] + width / 2, 66.0, "N/A", ha="center", va="bottom", fontsize=8, color=FG_MID)

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Zero-shot baselines on the full and dental-only test sets")
    ax.set_ylim(64, 90)
    ax.legend(loc="upper left")
    style_axis(ax)
    save(fig, "fig_4_3_full_vs_dental_baseline.png")


def plot_full_data_distill_results():
    configs = ["7B Stage 1", "7B Stage 2", "14B Stage 1", "14B + Llama"]
    full_mean = np.array([85.60, 85.20, 88.67, 87.25])
    full_best = np.array([86.28, 85.77, 89.10, 87.59])
    dental_mean = np.array([73.60, 73.33, 79.20, 80.00])

    x = np.arange(len(configs))
    width = 0.24

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.bar(x - width, full_mean, width, label="Full mean", color=FG_LIGHT, edgecolor=FG_DARK, hatch="//")
    ax.bar(x, full_best, width, label="Full best", color=FG_MID, edgecolor=FG_DARK)
    ax.bar(x + width, dental_mean, width, label="Dental mean", color="white", edgecolor=FG_DARK, hatch="..")

    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Distillation results under the full-data resplit setting")
    ax.set_ylim(70, 91)
    ax.legend(ncol=3, loc="upper center")
    style_axis(ax)
    save(fig, "fig_4_4_full_data_distill_results.png")


def plot_overall_ranking():
    methods = [
        "14B Stage1 best",
        "14B Stage1 mean",
        "DeepSeek teacher",
        "14B Llama mean",
        "7B Stage1 best",
        "7B Stage1 mean",
        "14B zero-shot",
        "7B zero-shot",
        "Llama teacher",
    ]
    scores = np.array([89.10, 88.67, 87.18, 87.25, 86.28, 85.60, 83.55, 76.49, 72.45])
    gains = ["+5.55pp", "+5.12pp", "-", "+3.70pp", "+9.79pp", "+9.11pp", "-", "-", "-"]

    order = np.argsort(scores)
    methods = [methods[i] for i in order]
    scores = scores[order]
    gains = [gains[i] for i in order]

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    colors = [FG_LIGHT if "teacher" in m.lower() or "zero-shot" in m.lower() else FG_MID for m in methods]
    bars = ax.barh(methods, scores, color=colors)

    for bar, score, gain in zip(bars, scores, gains):
        ax.text(score + 0.25, bar.get_y() + bar.get_height() / 2, f"{score:.2f}%  {gain}", va="center")

    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Representative overall results ranking")
    ax.set_xlim(70, 91.5)
    style_axis(ax)
    save(fig, "fig_4_5_overall_ranking.png")


if __name__ == "__main__":
    plot_teacher_quality_vs_gain()
    plot_single_teacher_results()
    plot_full_vs_dental_baseline()
    plot_full_data_distill_results()
    plot_overall_ranking()
    print(f"Generated figures in {OUT_DIR}")
