#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""三组 α 消融 canonical 数据落盘 + 画头条核心规律图 + 3张分组α曲线图。Module 21。"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 中文字体 (Noto CJK) — 注册后实际 family name 是 'Noto Sans CJK JP'
_FAM = None
for fp in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
           "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"]:
    if os.path.exists(fp):
        font_manager.fontManager.addfont(fp)
        if _FAM is None:
            _FAM = font_manager.FontProperties(fname=fp).get_name()
plt.rcParams["font.family"] = _FAM or "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

ALPHAS = [0.0, 0.15, 0.25, 0.35, 0.50, 0.65, 1.0]

# canonical full(991) 3-seed 均值
FULL = {
    "①DeepSeek→14B": [89.14, 89.17, 89.10, 88.67, 88.67, 88.33, 87.55],
    "②DeepSeek→7B":  [85.74, 85.87, 85.57, 85.67, 85.57, 85.30, 84.60],
    "③Llama→14B":    [89.44, 88.93, 89.17, 88.53, 88.06, 86.82, 81.10],
}
# canonical dental(125)
DENTAL = {
    "①DeepSeek→14B": [82.13, 81.07, 80.53, 79.20, 80.00, 77.07, 77.33],
    "②DeepSeek→7B":  [72.80, 73.87, 74.67, 74.93, 75.20, 73.87, 72.27],
    "③Llama→14B":    [82.93, 82.40, 83.47, 81.33, 80.80, 80.53, 77.33],
}
META = {
    "①DeepSeek→14B": dict(teacher="DeepSeek-V3", student="Qwen2.5-14B", mismatch=12.2, opt_alpha=0.15, note="0~0.25平台区,α=0.15/0=并列最优"),
    "②DeepSeek→7B":  dict(teacher="DeepSeek-V3", student="Qwen2.5-7B",  mismatch=12.2, opt_alpha=0.15, note="最优α右移至0.15(弱学生需软标签正则)"),
    "③Llama→14B":    dict(teacher="Llama-3.3-70B", student="Qwen2.5-14B", mismatch=48.4, opt_alpha=0.0, note="α=0最强烈最优,纯KL暴跌-8.3pp"),
}

os.makedirs("21_ablation/figs", exist_ok=True)
json.dump({"alphas":ALPHAS,"full":FULL,"dental":DENTAL,"meta":META},
          open("21_ablation/figs/ablation_data.json","w"), ensure_ascii=False, indent=2)

colors = {"①DeepSeek→14B":"#2c7fb8", "②DeepSeek→7B":"#41ab5d", "③Llama→14B":"#e6550d"}
markers = {"①DeepSeek→14B":"o", "②DeepSeek→7B":"s", "③Llama→14B":"^"}

# ===== 头条核心规律图: 三曲线叠一张 =====
fig, ax = plt.subplots(figsize=(8.5, 5.6))
for name, ys in FULL.items():
    ax.plot(ALPHAS, ys, marker=markers[name], color=colors[name], lw=2.2, ms=8,
            label=f"{name} (教师不一致{META[name]['mismatch']}%)")
    # 标注每组最优点
    oa = META[name]["opt_alpha"]; oi = ALPHAS.index(oa)
    ax.scatter([oa],[ys[oi]], s=180, facecolors="none", edgecolors=colors[name], lw=2.5, zorder=5)
ax.set_xlabel("α  (KL 蒸馏权重;  Loss = α·KL + (1−α)·CE)", fontsize=12)
ax.set_ylabel("Canonical 准确率 (%, full 991, 3-seed 均值)", fontsize=12)
ax.set_title("最优 α 随教师质量与学生容量的移动规律\n(空心圈=各组最优 α;  纯 SFT 端 α=0 在左, 纯模仿教师端 α=1 在右)", fontsize=12.5, pad=12)
ax.set_xticks(ALPHAS)
ax.grid(alpha=0.3, ls="--")
ax.legend(fontsize=10.5, loc="lower left")
# 两个维度规律的注释
ax.annotate("教师越差(③48.4%)→纯KL越崩塌\n(③ α=1.0 暴跌 -8.3pp)",
            xy=(1.0, 81.10), xytext=(0.55, 82.5), fontsize=9.5, color="#e6550d",
            arrowprops=dict(arrowstyle="->", color="#e6550d", lw=1.5))
ax.annotate("强教师下纯模仿仅小幅下降\n(①-1.6pp / ②-1.3pp)",
            xy=(1.0, 87.55), xytext=(0.42, 86.3), fontsize=9.5, color="#2c7fb8",
            arrowprops=dict(arrowstyle="->", color="#2c7fb8", lw=1.3))
fig.tight_layout()
fig.savefig("21_ablation/figs/fig1_headline_regularity.png", dpi=150)
plt.close(fig)

# ===== 3张分组α曲线图 (full + dental 双线) =====
for idx, (name, ys) in enumerate(FULL.items(), 1):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(ALPHAS, ys, marker="o", color=colors[name], lw=2.2, ms=7, label="full (991)")
    d = DENTAL[name]
    if all(v is not None for v in d):
        ax.plot(ALPHAS, d, marker="s", color=colors[name], lw=1.6, ms=6, ls="--", alpha=0.6, label="dental (125)")
    oa = META[name]["opt_alpha"]; oi = ALPHAS.index(oa)
    ax.scatter([oa],[ys[oi]], s=170, facecolors="none", edgecolors="red", lw=2.2, zorder=5, label=f"最优 α={oa}")
    m = META[name]
    ax.set_title(f"组合{name}\n教师 {m['teacher']} → 学生 {m['student']}  (不一致率 {m['mismatch']}%)", fontsize=11)
    ax.set_xlabel("α (KL 权重)", fontsize=11)
    ax.set_ylabel("Canonical 准确率 (%)", fontsize=11)
    ax.set_xticks(ALPHAS)
    ax.grid(alpha=0.3, ls="--")
    ax.legend(fontsize=9.5)
    fig.tight_layout()
    fig.savefig(f"21_ablation/figs/fig{idx+1}_group{idx}.png", dpi=150)
    plt.close(fig)

print("图已生成:")
for f in sorted(os.listdir("21_ablation/figs")):
    print("  21_ablation/figs/"+f)
