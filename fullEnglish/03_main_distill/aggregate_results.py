#!/usr/bin/env python3
"""fullEnglish — 聚合主实验结果 -> runs/RESULTS.md.

对齐「学生 vs 教师」同题集口径:
  - 教师: labels/teacher_test_{medqa,medmcqa,mmlu}.jsonl (zero-shot, 同题)
  - 学生: runs/eval_results.json (zeroshot + 各 adapter)
头条问题: 全医科英文数据下, 学生(Qwen2.5-32B, α=0) 能否超越教师.
"""
import json
import os
import glob
import numpy as np

FE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(FE, "runs")
LABELS = os.path.join(FE, "labels")

MCQ_SETS = ["test_medqa", "test_medmcqa", "test_mmlu"]
ALL_SETS = MCQ_SETS + ["test_pubmedqa"]
SET_COUNTS = {"test_medqa": 1273, "test_medmcqa": 4183, "test_mmlu": 2837, "test_pubmedqa": 1000}


def teacher_acc(set_name):
    p = os.path.join(LABELS, f"teacher_{set_name}.jsonl")
    if not os.path.exists(p):
        return None
    n = c = 0
    for line in open(p):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        gt = str(r.get("OriginalAnswer") or r.get("Answer", "")).strip().upper()
        ta = str(r.get("TeacherAnswer") or r.get("Answer", "")).strip().upper()
        if gt in "ABCDE" and ta in "ABCDE":
            n += 1
            c += int(ta == gt)
    return round(100 * c / n, 2) if n else None


def main():
    ev_path = os.path.join(RUN, "eval_results.json")
    if not os.path.exists(ev_path):
        print("[no eval_results.json] 先跑 evaluate_all.py / run_main_distill.sh")
        raise SystemExit(0)
    ev = json.load(open(ev_path))
    zeroshot = ev.get("zeroshot", {})
    adapters = ev.get("adapters", {})

    # 教师候选名 (从 teacher_candidate.json)
    cand = json.load(open(os.path.join(FE, "teacher_candidate.json"))) if os.path.exists(os.path.join(FE, "teacher_candidate.json")) else {}
    teacher_name = os.environ.get("TEACHER_NAME", cand.get("name", "Teacher"))

    lines = ["# fullEnglish — 主蒸馏结果 (学生 Qwen2.5-32B)\n",
             f"教师: **{teacher_name}** (zero-shot, 同题集)  |  学生: Qwen2.5-32B-Instruct\n",
             "头条: **α=0 (纯 GT SFT / 决策空间监督)** × 3 seed\n"]

    # 分 arm 聚合 (a00 / a35 / a10)
    arms = {}
    for name, info in adapters.items():
        # name 形如 32B_a00_s11
        parts = name.split("_")
        if len(parts) >= 2:
            arm = parts[1]
            arms.setdefault(arm, []).append(info["acc"])

    # 表格: 每测试集, 教师 / 学生零样本 / 学生α0(mean±std) / delta
    lines.append("## 分测试集: 教师 vs 学生 (同题集)\n")
    lines.append("| 测试集 | 教师 | 学生零样本 | 学生α=0 (mean±std) | Δ(学生-教师) |")
    lines.append("|---|---|---|---|---|")

    def arm_stat(arm, set_name):
        accs = [a[set_name] for a in arms.get(arm, []) if set_name in a]
        if not accs:
            return None
        return round(np.mean(accs), 2), round(np.std(accs), 2), len(accs)

    combined_t = 0
    combined_s = 0
    combined_n = 0
    for s in ALL_SETS:
        t = teacher_acc(s)
        z = zeroshot.get(s)
        st = arm_stat("a00", s)
        row = f"| {s} | {t if t is not None else '—'} | {z if z is not None else '—'} | "
        if st:
            row += f"{st[0]}±{st[1]} (n={st[2]}) | {round(st[0]-t,2) if t is not None else '—'} |"
        else:
            row += "— | — |"
        lines.append(row)
        if s in MCQ_SETS and t is not None and st:
            combined_t += t * SET_COUNTS[s]
            combined_s += st[0] * SET_COUNTS[s]
            combined_n += SET_COUNTS[s]

    # 组合 MCQ (加权)
    if combined_n:
        ct = round(combined_t / combined_n, 2)
        cs = round(combined_s / combined_n, 2)
        lines.append("")
        lines.append("## 组合 MCQ (MedQA+MedMCQA+MMLU 加权)\n")
        lines.append(f"- 教师: **{ct}%**   学生α=0: **{cs}%**   Δ = **{round(cs-ct,2):+.2f}pp**")
        if cs > ct:
            lines.append(f"- ✅ **学生超越教师** (+{round(cs-ct,2)}pp) —— 复刻中文全科结论 (中文 14B 89.14% > 教师 87.18%).")
        else:
            lines.append(f"- ❌ 学生未超越教师 ({round(cs-ct,2):+.2f}pp). 参照英文牙科 (headroom≈20pp 未超越) 的机制分析, 如实报告.")

    # α 扫描 (若存在)
    if len(arms) > 1:
        lines.append("")
        lines.append("## α 扫描 (复现「KL 越多越差」跨语言结论)\n")
        lines.append("| arm | test_medqa | test_medmcqa | test_mmlu | test_pubmedqa |")
        lines.append("|---|---|---|---|---|")
        for arm in ["a00", "a35", "a10"]:
            if arm not in arms:
                continue
            alpha = {"a00": 0.0, "a35": 0.35, "a10": 1.0}[arm]
            cells = []
            for s in ALL_SETS:
                st = arm_stat(arm, s)
                cells.append(f"{st[0]}±{st[1]}" if st else "—")
            lines.append(f"| α={alpha} | " + " | ".join(cells) + " |")

    # 结论
    lines.append("")
    lines.append("## 结论")
    lines.append("- 头条: α=0 学生 vs 教师, 见上「组合 MCQ」Δ.")
    lines.append("- α 单调性: 若 α=0 ≥ α=0.35 ≥ α=1.0, 则英文全科同样复制中文「决策空间监督优于 KL 蒸馏」.")
    lines.append("- PubMedQA 是 held-out 判断题, 蒸馏训练未用; 若学生在 PubMedQA 接近/超过教师, 说明迁移的是医学推理而非背题.")

    out = os.path.join(RUN, "RESULTS.md")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
