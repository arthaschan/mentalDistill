#!/usr/bin/env python3
"""生成论文图 1（headroom 相变散点图）与图 2（四学生单调线）。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 中文字体
for name in ["Noto Sans CJK SC", "Noto Sans CJK JP"]:
    if any(f.name == name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
        break
plt.rcParams["axes.unicode_minus"] = False

import os
os.makedirs("fullEnglish/figures", exist_ok=True)

# ============ 图 1：headroom 相变散点图 ============
# (headroom, 超越幅度, 标签)
success = [
    (3.63, 1.49, "中文全科\n(DeepSeek)"),
    (1.60, 3.20, "中文牙科\n(DeepSeek)"),
    (2.46, 1.80, "英文全科·弱教师\n(Qwen2.5-32B)"),
    (1.41, 2.43, "英文全科·弱教师\n(Llama-70B)"),
    (1.23, 2.89, "英文牙科·弱教师\n(Qwen2.5-32B)"),
    (0.82, 3.36, "英文牙科·弱教师\n(Llama-70B)"),
]
fail = [
    (5.96, -2.21, "英文全科\nQwen3-32B"),
    (7.37, -3.53, "英文全科\nLlama-70B"),
    (8.42, -4.16, "英文全科\nQwen2.5-32B"),
    (11.97, -7.39, "英文全科\nQwen2.5-14B"),
]

fig, ax = plt.subplots(figsize=(8, 5.5))
for h, d, lab in success:
    ax.scatter(h, d, s=90, c="#2ca02c", marker="o", zorder=3)
    ax.annotate(lab, (h, d), textcoords="offset points", xytext=(6, -4), fontsize=8)
for h, d, lab in fail:
    ax.scatter(h, d, s=90, c="#d62728", marker="s", zorder=3)
    ax.annotate(lab, (h, d), textcoords="offset points", xytext=(6, -6), fontsize=8)

ax.axhline(0, color="gray", ls="--", lw=1)
ax.axvline(4.2, color="gray", ls=":", lw=1)
ax.fill_betweenx([-9, 5], 0, 4.2, color="#2ca02c", alpha=0.08)
ax.set_xlabel("领先幅度 headroom（教师零样本 − 学生零样本，百分点）")
ax.set_ylabel("超越幅度（学生训练后 − 教师零样本，百分点）")
ax.set_title("图 1　蒸馏「学生超越教师」的相变：headroom < 增益(~4pp) 则超越")
ax.text(2.0, 4.4, "超越区（headroom < 增益）", fontsize=9, color="#2ca02c", ha="center")
ax.text(9.0, 4.4, "未超越区（headroom > 增益）", fontsize=9, color="#d62728", ha="center")
ax.set_xlim(0, 13.5)
ax.set_ylim(-8.5, 5)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fullEnglish/figures/fig1_headroom_phase.png", dpi=200)
plt.close()
print("已生成 fullEnglish/figures/fig1_headroom_phase.png")

# ============ 图 2：四学生单调线 ============
students = [
    (67.83, -7.39, "Qwen2.5-14B"),
    (71.38, -4.16, "Qwen2.5-32B"),
    (72.43, -3.53, "Llama-70B"),
    (73.84, -2.21, "Qwen3-32B"),
]
xs = [s[0] for s in students]
ys = [s[1] for s in students]
labs = [s[2] for s in students]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(xs, ys, marker="o", ms=8, lw=2, color="#1f77b4")
for x, y, lab in zip(xs, ys, labs):
    ax.annotate(lab, (x, y), textcoords="offset points", xytext=(6, 8), fontsize=9)
ax.axhline(0, color="#d62728", ls="--", lw=1.2)
ax.text(73.9, 0.4, "超越线（y=0）", fontsize=9, color="#d62728", ha="right")
ax.set_xlabel("学生零样本准确率（%，英文全科 8293 题）")
ax.set_ylabel("距教师差距（学生训练后 − flash 79.80%，百分点）")
ax.set_title("图 2　英文全科四学生单调线：学生越强越接近教师，但始终未超越")
ax.set_ylim(-9, 2)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fullEnglish/figures/fig2_four_students.png", dpi=200)
plt.close()
print("已生成 fullEnglish/figures/fig2_four_students.png")
