#!/usr/bin/env python3
"""聚合教师预评估结果 -> 教师能力先验表 + 学生 headroom 分析.

扫描:
  labels/*.jsonl       (API 硬标签: OriginalAnswer=GT, TeacherAnswer=预测)
  logprobs/*_logprobs.jsonl  (本地真实分布: TeacherDist + TeacherAnswer + OriginalAnswer)

产出:
  reports/teacher_prior.md  (排序先验表 + 每源准确率 + headroom)
  reports/screening.json    (结构化, 供融合 oracle 复用)
"""
import json
import os
import glob
import numpy as np
from collections import defaultdict

FE = os.path.dirname(os.path.abspath(__file__))
LETTERS = ["A", "B", "C", "D", "E"]
STUDENT_NAME = os.environ.get("STUDENT_NAME", "Qwen32B")  # 学生 base = Qwen2.5-32B


def load_rows(path):
    rows = {}
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            gt = str(r.get("OriginalAnswer") or r.get("Answer", "")).strip().upper()
            pred = str(r.get("TeacherAnswer") or r.get("Answer", "")).strip().upper()
            if gt not in LETTERS or pred not in LETTERS:
                continue
            uid = r.get("uid") or r.get("Question", "")[:40]
            dist = r.get("TeacherDist", {})
            ent = None
            if isinstance(dist, dict) and dist:
                raw = np.array([float(dist.get(c, 0.0)) for c in LETTERS])
                s = raw.sum()
                if s > 1e-9:
                    raw = raw / s
                    ent = float(-np.sum(np.clip(raw, 1e-12, None) * np.log(np.clip(raw, 1e-12, None))))
            rows[uid] = {
                "gt": gt, "pred": pred, "correct": int(pred == gt),
                "ent": ent, "src": r.get("source", "?"), "subj": r.get("subject", "?"),
                "dist": raw.tolist() if ent is not None else None,
            }
    return rows


def main():
    teachers = {}
    for f in sorted(glob.glob(os.path.join(FE, "labels", "*.jsonl"))):
        name = os.path.basename(f).replace(".jsonl", "")
        rows = load_rows(f)
        if rows:
            teachers[name] = rows
    for f in sorted(glob.glob(os.path.join(FE, "logprobs", "*_logprobs.jsonl"))):
        name = os.path.basename(f).replace("_logprobs.jsonl", "")
        rows = load_rows(f)
        if rows:
            teachers[name] = rows

    if not teachers:
        print("[no teacher labels yet] 先运行 run_screening.sh")
        raise SystemExit(0)

    # overall + per-source acc
    prior = {}
    per_src = {}
    for name, rows in teachers.items():
        acc = 100 * np.mean([v["correct"] for v in rows.values()])
        ents = [v["ent"] for v in rows.values() if v["ent"] is not None]
        prior[name] = {"n": len(rows), "acc": round(acc, 2),
                       "mean_ent": round(float(np.mean(ents)), 4) if ents else None}
        by_src = defaultdict(list)
        for v in rows.values():
            by_src[v["src"]].append(v["correct"])
        per_src[name] = {s: round(100 * np.mean(v), 1) for s, v in sorted(by_src.items())}

    order = sorted(prior, key=lambda k: -prior[k]["acc"])
    best = order[0]

    # 学生 headroom: 最强教师 - 学生零样本地板
    headroom = None
    student_acc = None
    if STUDENT_NAME in prior:
        student_acc = prior[STUDENT_NAME]["acc"]
        headroom = round(prior[best]["acc"] - student_acc, 2)

    # 每源最强教师
    sources = sorted({s for m in per_src.values() for s in m})
    per_src_winner = {}
    for s in sources:
        vals = {n: per_src[n].get(s) for n in teachers if per_src[n].get(s) is not None}
        if vals:
            w = max(vals, key=vals.get)
            per_src_winner[s] = (w, vals[w])

    os.makedirs(os.path.join(FE, "reports"), exist_ok=True)
    out = {
        "teacher_prior": prior, "order": order, "best_teacher": best,
        "per_source_acc": per_src, "per_source_winner": per_src_winner,
        "student_name": STUDENT_NAME, "student_zero_shot_acc": student_acc,
        "headroom_pp": headroom,
    }
    with open(os.path.join(FE, "reports", "screening.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # markdown
    md = ["# fullEnglish — Teacher Screening (教师预评估)\n",
          f"Pool: {prior[order[0]]['n']} items (MedQA/MedMCQA/MMLU 各 200). Teachers: {len(teachers)}.\n",
          "## Teacher prior (zero-shot, English prompt)\n",
          "| rank | teacher | acc% | mean_entropy |", "|---|---|---|---|"]
    for i, n in enumerate(order, 1):
        e = f"{prior[n]['mean_ent']}" if prior[n]["mean_ent"] is not None else "—"
        tag = "  ⬅ 学生 base" if n == STUDENT_NAME else ""
        md.append(f"| {i} | {n}{tag} | {prior[n]['acc']} | {e} |")

    md.append("\n## Per-source accuracy")
    hdr = "| teacher | " + " | ".join(sources) + " |"
    md.append(hdr)
    md.append("|---|" + "|".join(["---"] * len(sources)) + "|")
    for n in order:
        md.append("| " + n + " | " + " | ".join(str(per_src[n].get(s, "")) for s in sources) + " |")

    md.append("\n## Headroom (能否超越教师的关键判据)")
    if headroom is not None:
        md.append(f"- 最强教师 **{best} = {prior[best]['acc']}%**")
        md.append(f"- 学生零样本地板 **{STUDENT_NAME} = {student_acc}%**")
        md.append(f"- **headroom = {headroom:+.2f}pp**")
        if headroom <= 8:
            md.append(f"- => headroom 较小 (≤8pp), 参照中文 (headroom≈3.6pp 时 14B 学生超越), "
                      f"Choice-Head 蒸馏有望追平/超越. 建议直接进主实验 (α=0 头条).")
        else:
            md.append(f"- => headroom 较大 (>{8}pp), 单靠决策空间监督难填平; 主实验仍跑, 但预期可能不超越, "
                      f"如实报告, 参考英文牙科 (headroom≈20pp 未超越) 的机制分析.")
    else:
        md.append(f"- 未找到学生 {STUDENT_NAME} 的 zero-shot, 请确认 run_screening.sh 已跑 Qwen32B.")
    md.append("\n### Per-source winners")
    md.append("| source | winner | acc% |")
    md.append("|---|---|---|")
    for s, (w, a) in sorted(per_src_winner.items()):
        md.append(f"| {s} | {w} | {a} |")
    open(os.path.join(FE, "reports", "teacher_prior.md"), "w").write("\n".join(md))

    print("=== TEACHER PRIOR (fullEnglish, zero-shot) ===")
    for i, n in enumerate(order, 1):
        tag = "  <== 学生base" if n == STUDENT_NAME else ""
        print(f"  {i}. {n:14s} {prior[n]['acc']:6.2f}%  ent={prior[n]['mean_ent']}{tag}")
    if headroom is not None:
        print(f"最强教师 {best}={prior[best]['acc']}%  vs  学生 {STUDENT_NAME}={student_acc}%  headroom={headroom:+.2f}pp")
    print(f"-> reports/teacher_prior.md, reports/screening.json")


if __name__ == "__main__":
    main()
