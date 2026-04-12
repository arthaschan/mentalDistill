#!/usr/bin/env python3
"""
merge_teacher_soft_labels.py — 多老师软标签融合

将 K 个老师的 TeacherDist 按全局权重加权融合，输出单一融合标签文件。
融合公式: p_ens(x) = Σ w_i · p_i(x)
"""
import argparse
import json
import sys
from pathlib import Path

OPTION_LETTERS = ["A", "B", "C", "D", "E"]


def load_teacher_labels(path: str):
    """加载 teacher soft label 文件，返回 {question_text: row} 映射"""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_dist(row):
    """从行中提取归一化的 TeacherDist"""
    raw = row.get("TeacherDist", {})
    probs = {}
    total = 0.0
    for ch in OPTION_LETTERS:
        v = float(raw.get(ch, 0.0))
        probs[ch] = max(v, 0.0)
        total += probs[ch]
    if total <= 0:
        return None
    return {ch: probs[ch] / total for ch in OPTION_LETTERS}


def merge_distributions(teacher_dists, weights):
    """加权融合多个分布"""
    merged = {ch: 0.0 for ch in OPTION_LETTERS}
    w_sum = 0.0
    for dist, w in zip(teacher_dists, weights):
        if dist is None:
            continue
        for ch in OPTION_LETTERS:
            merged[ch] += w * dist[ch]
        w_sum += w
    if w_sum <= 0:
        return None
    return {ch: merged[ch] / w_sum for ch in OPTION_LETTERS}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True,
                   help="Base data file (train.jsonl) for question alignment")
    p.add_argument("--teachers", nargs="+", required=True,
                   help="Teacher soft label files (teacher_train_soft.jsonl)")
    p.add_argument("--weights", nargs="+", type=float, required=True,
                   help="Teacher weights (same order as --teachers)")
    p.add_argument("--output", required=True,
                   help="Output merged soft label file")
    p.add_argument("--teacher_names", nargs="+", default=None,
                   help="Teacher names for logging")
    args = p.parse_args()

    assert len(args.teachers) == len(args.weights), \
        f"teachers ({len(args.teachers)}) and weights ({len(args.weights)}) must match"

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    names = args.teacher_names or [f"T{i}" for i in range(len(args.teachers))]

    # 加载 base data
    base_rows = load_teacher_labels(args.base)
    print(f"[MERGE] Base data: {len(base_rows)} samples from {args.base}")

    # 加载所有 teacher 标签, 按 Question 文本索引
    teacher_maps = []
    for i, tp in enumerate(args.teachers):
        rows = load_teacher_labels(tp)
        qmap = {}
        for r in rows:
            q = r.get("Question", "").strip()
            if q:
                qmap[q] = r
        teacher_maps.append(qmap)
        print(f"[MERGE] Teacher {names[i]}: {len(rows)} rows, {len(qmap)} unique questions")

    # 融合
    total = 0
    merged_count = 0
    missing_teachers = [0] * len(args.teachers)
    agreement_stats = {"all_agree": 0, "majority": 0, "split": 0}

    with open(args.output, "w", encoding="utf-8") as wf:
        for row in base_rows:
            q = row.get("Question", "").strip()
            total += 1

            dists = []
            active_weights = []
            teacher_answers = []
            for i, qmap in enumerate(teacher_maps):
                trow = qmap.get(q)
                if trow is None:
                    missing_teachers[i] += 1
                    dists.append(None)
                    active_weights.append(0.0)
                    teacher_answers.append(None)
                else:
                    d = get_dist(trow)
                    dists.append(d)
                    active_weights.append(args.weights[i])
                    ta = trow.get("TeacherAnswer", "").strip().upper()
                    teacher_answers.append(ta if ta in OPTION_LETTERS else None)

            merged = merge_distributions(dists, active_weights)

            # 分歧统计
            valid_answers = [a for a in teacher_answers if a]
            if valid_answers:
                from collections import Counter
                counts = Counter(valid_answers)
                top_count = counts.most_common(1)[0][1]
                if top_count == len(valid_answers):
                    agreement_stats["all_agree"] += 1
                elif top_count > len(valid_answers) / 2:
                    agreement_stats["majority"] += 1
                else:
                    agreement_stats["split"] += 1

            out_row = dict(row)
            if merged is not None:
                out_row["TeacherDist"] = merged
                # 融合后的 hard prediction
                best_ch = max(merged, key=merged.get)
                out_row["TeacherAnswer"] = best_ch
                merged_count += 1
            else:
                # 回退到 GT
                gt = row.get("Answer", "A").strip().upper()
                out_row["TeacherDist"] = {ch: (1.0 if ch == gt else 0.0) for ch in OPTION_LETTERS}
                out_row["TeacherAnswer"] = gt

            # 记录各老师的原始答案 (用于后续分析)
            out_row["TeacherAnswers"] = {
                names[i]: teacher_answers[i] for i in range(len(names))
            }

            wf.write(json.dumps(out_row, ensure_ascii=False) + "\n")

    # 报告
    print(f"\n{'='*50}")
    print(f"融合统计")
    print(f"{'='*50}")
    print(f"Total: {total}")
    print(f"Merged: {merged_count} ({merged_count/total*100:.1f}%)")
    print(f"Fallback to GT: {total - merged_count}")
    print(f"\nTeacher 缺失统计:")
    for i, name in enumerate(names):
        print(f"  {name}: {missing_teachers[i]} missing ({missing_teachers[i]/total*100:.1f}%)")
    print(f"\n老师一致性:")
    print(f"  全部一致: {agreement_stats['all_agree']} ({agreement_stats['all_agree']/total*100:.1f}%)")
    print(f"  多数一致: {agreement_stats['majority']} ({agreement_stats['majority']/total*100:.1f}%)")
    print(f"  严重分歧: {agreement_stats['split']} ({agreement_stats['split']/total*100:.1f}%)")

    # GT 对齐检查
    gt_match = 0
    for row in base_rows:
        q = row.get("Question", "").strip()
        gt = row.get("Answer", "").strip().upper()
        # 重新计算融合结果
        dists_check = []
        ws_check = []
        for i, qmap in enumerate(teacher_maps):
            trow = qmap.get(q)
            if trow:
                d = get_dist(trow)
                dists_check.append(d)
                ws_check.append(args.weights[i])
        m = merge_distributions(dists_check, ws_check)
        if m:
            pred = max(m, key=m.get)
            if pred == gt:
                gt_match += 1
    print(f"\n融合标签 vs GT 一致率: {gt_match}/{total} ({gt_match/total*100:.2f}%)")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
