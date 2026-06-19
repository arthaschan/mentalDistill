#!/usr/bin/env python3
"""
build_geometry_filtered_dataset.py — 任务 A：按几何可蒸馏性分数过滤蒸馏样本

输入：已构造的 train_head_distill.jsonl（含 clean_teacher 行的真实 TeacherDist）。
对每个 clean_teacher 行计算 GT-无关的可蒸馏性分数 s(p)（与 combined_predictor 同款
标准化逻辑回归方向：低熵/大体积元/大 margin = 高分），然后产出三个训练集变体：

  baseline_all   : 保持原样（所有 clean_teacher 行带 KL 监督）
  geom_topK      : 仅分数最高的 K% 行保留 KL 监督；其余降级为 GT-only（去掉 SelectiveSource）
  random_topK    : 随机 K% 行保留 KL 监督（对照组，隔离「几何」与「单纯减量」）

KL 监督开关 = 训练脚本里的 SelectiveSource=="clean_teacher"（line 139）。
降级 = 删除 TeacherDist/SelectiveSource，行仍保留用于 GT CE。

所有变体的总样本数、GT 标签完全相同，唯一差异是「哪些样本带教师 KL」。
"""
import argparse
import json
import math
import os
import random

OPTION_LETTERS = ["A", "B", "C", "D", "E"]


def dist_vec(row):
    raw = row.get("TeacherDist")
    if not isinstance(raw, dict):
        return None
    v = [float(raw.get(k, 0.0)) for k in OPTION_LETTERS]
    if sum(v) <= 1e-9:
        return None
    s = sum(max(x, 0.0) for x in v)
    return [max(x, 1e-12) / s for x in v]


def confidence_score(p):
    """GT-independent distillability score. Higher = sharper/more reliable teacher.
    Composite matching combined_predictor signs: -entropy dominant, +logdet_g, +margin."""
    srt = sorted(p, reverse=True)
    H = -sum(x * math.log(x + 1e-12) for x in p)
    logdet = -0.5 * sum(math.log10(x) for x in p)
    margin = srt[0] - srt[1]
    # standardized-ish composite; weights mirror combined_predictor (entropy strongest)
    return (-0.253 * H) + (0.226 * (logdet / 10.0)) + (0.102 * margin) + (0.093 * srt[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="16_llama70b_choice_head/data/train_head_distill.jsonl")
    ap.add_argument("--outdir", default="research/distillability/datasets")
    ap.add_argument("--keep_frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rows = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]

    clean_idx = []
    scores = {}
    for i, r in enumerate(rows):
        if str(r.get("SelectiveSource", "")) == "clean_teacher":
            p = dist_vec(r)
            if p is not None:
                clean_idx.append(i)
                scores[i] = confidence_score(p)

    n_clean = len(clean_idx)
    keep_n = int(round(args.keep_frac * n_clean))
    print(f"clean_teacher rows: {n_clean}, keep_n (top {args.keep_frac:.0%}): {keep_n}")

    # geometry-ranked keep set
    geom_sorted = sorted(clean_idx, key=lambda i: -scores[i])
    geom_keep = set(geom_sorted[:keep_n])

    # random keep set (control)
    rng = random.Random(args.seed)
    rand_keep = set(rng.sample(clean_idx, keep_n))

    def demote(row):
        r = dict(row)
        r.pop("TeacherDist", None)
        r.pop("SelectiveSource", None)
        r.pop("TeacherAnswer", None)
        return r

    def write_variant(name, keep_set):
        path = os.path.join(args.outdir, f"train_{name}.jsonl")
        kept = 0
        with open(path, "w", encoding="utf-8") as f:
            for i, r in enumerate(rows):
                if i in scores:  # a clean_teacher row
                    if i in keep_set:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                        kept += 1
                    else:
                        f.write(json.dumps(demote(r), ensure_ascii=False) + "\n")
                else:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  [{name}] total={len(rows)} kept_KL={kept} -> {path}")
        return path, kept

    # baseline keeps all clean
    write_variant("baseline_all", set(clean_idx))
    write_variant(f"geom_top{int(args.keep_frac*100)}", geom_keep)
    write_variant(f"random_top{int(args.keep_frac*100)}", rand_keep)

    # diagnostics: teacher-correctness rate within each keep set (uses GT, analysis only)
    def correct_rate(keep_set):
        c = 0
        for i in keep_set:
            r = rows[i]
            gt = str(r.get("OriginalAnswer") or r.get("Answer", "")).strip().upper()
            p = dist_vec(r)
            if p is None:
                continue
            pred = OPTION_LETTERS[p.index(max(p))]
            if pred == gt:
                c += 1
        return 100.0 * c / len(keep_set) if keep_set else 0.0

    print(f"\nTeacher-correctness within keep sets (diagnostic, uses GT):")
    print(f"  all clean    : {correct_rate(set(clean_idx)):.2f}%  (n={n_clean})")
    print(f"  geom_top     : {correct_rate(geom_keep):.2f}%  (n={len(geom_keep)})")
    print(f"  random_top   : {correct_rate(rand_keep):.2f}%  (n={len(rand_keep)})")


if __name__ == "__main__":
    main()
