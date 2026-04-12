#!/usr/bin/env python3
"""
fuse_multivote_teachers.py — 基于多次投票的真实软标签融合 + 自一致性过滤

输入: 每个教师的 multivote JSONL (含 TeacherVotes, TeacherDist)
输出: 融合后的训练数据 JSONL

自一致性过滤策略:
  - consistency = max(vote_counts) / total_votes
  - consistency >= high_thresh  → 全权重
  - high_thresh > consistency >= low_thresh → 半权重 (downweight_factor)
  - consistency < low_thresh → 排除该教师对该样本的贡献
  - 如果所有教师都被排除 → 回退到 GT one-hot 标签
"""
import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

OPTION_LETTERS = ["A", "B", "C", "D", "E"]


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_dist(row):
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


def compute_consistency(votes):
    """从投票列表计算自一致性 = max_vote_count / total"""
    if not votes:
        return 0.0
    counts = Counter(votes)
    return counts.most_common(1)[0][1] / len(votes)


def entropy(dist):
    return -sum(v * math.log(v + 1e-12) for v in dist.values())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True,
                   help="GT data (train.jsonl)")
    p.add_argument("--teachers", nargs="+", required=True,
                   help="Multi-vote teacher JSONL files")
    p.add_argument("--weights", nargs="+", type=float, required=True,
                   help="Base teacher weights (accuracy-based)")
    p.add_argument("--teacher_names", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--high_thresh", type=float, default=0.78,
                   help="Consistency >= this → full weight (default: 7/9≈0.78)")
    p.add_argument("--low_thresh", type=float, default=0.56,
                   help="Consistency >= this → downweight (default: 5/9≈0.56)")
    p.add_argument("--downweight_factor", type=float, default=0.5,
                   help="Weight multiplier for mid-consistency samples")
    p.add_argument("--report", default=None,
                   help="Path to write JSON report")
    args = p.parse_args()

    assert len(args.teachers) == len(args.weights) == len(args.teacher_names)

    # 加载 base
    base_rows = load_jsonl(args.base)
    print(f"[FUSE] Base: {len(base_rows)} samples")

    # 加载教师标签，按 Question 索引
    teacher_maps = []
    for i, tp in enumerate(args.teachers):
        rows = load_jsonl(tp)
        qmap = {}
        for r in rows:
            q = r.get("Question", "").strip()
            if q:
                qmap[q] = r
        teacher_maps.append(qmap)
        print(f"[FUSE] Teacher {args.teacher_names[i]}: {len(qmap)} samples")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 统计
    stats = {
        "total": 0, "fused": 0, "gt_fallback": 0,
        "high_consistency": [0] * len(args.teachers),
        "mid_consistency": [0] * len(args.teachers),
        "low_consistency": [0] * len(args.teachers),
        "missing": [0] * len(args.teachers),
        "entropy_vals": [], "consistency_vals": [],
    }

    with open(args.output, "w", encoding="utf-8") as wf:
        for row in base_rows:
            q = row.get("Question", "").strip()
            gt = row.get("Answer", "A").strip().upper()
            stats["total"] += 1

            teacher_dists = []
            adjusted_weights = []
            teacher_answers = {}
            sample_consistencies = {}

            for i, qmap in enumerate(teacher_maps):
                trow = qmap.get(q)
                name = args.teacher_names[i]

                if trow is None:
                    stats["missing"][i] += 1
                    teacher_dists.append(None)
                    adjusted_weights.append(0.0)
                    continue

                dist = get_dist(trow)
                votes = trow.get("TeacherVotes", [])
                cons = compute_consistency(votes) if votes else 1.0
                sample_consistencies[name] = round(cons, 3)

                ta = trow.get("TeacherAnswer", "").strip().upper()
                if ta in OPTION_LETTERS:
                    teacher_answers[name] = ta

                # 根据自一致性调整权重
                if cons >= args.high_thresh:
                    w_mult = 1.0
                    stats["high_consistency"][i] += 1
                elif cons >= args.low_thresh:
                    w_mult = args.downweight_factor
                    stats["mid_consistency"][i] += 1
                else:
                    w_mult = 0.0
                    stats["low_consistency"][i] += 1

                teacher_dists.append(dist)
                adjusted_weights.append(args.weights[i] * w_mult)

            # 融合
            w_sum = sum(adjusted_weights)
            if w_sum > 0:
                merged = {ch: 0.0 for ch in OPTION_LETTERS}
                for dist, w in zip(teacher_dists, adjusted_weights):
                    if dist is not None and w > 0:
                        for ch in OPTION_LETTERS:
                            merged[ch] += w * dist[ch]
                merged = {ch: merged[ch] / w_sum for ch in OPTION_LETTERS}
                best_ch = max(OPTION_LETTERS, key=lambda k: merged[k])

                out_row = dict(row)
                out_row["TeacherDist"] = {ch: round(merged[ch], 6) for ch in OPTION_LETTERS}
                out_row["TeacherAnswer"] = best_ch
                out_row["TeacherAnswers"] = teacher_answers
                out_row["TeacherConsistency"] = sample_consistencies
                out_row["FusionSource"] = "multivote_fused"
                stats["fused"] += 1
                stats["entropy_vals"].append(entropy(merged))
                stats["consistency_vals"].extend(sample_consistencies.values())
            else:
                # 全部被排除 → GT fallback
                out_row = dict(row)
                out_row["TeacherDist"] = {ch: (1.0 if ch == gt else 0.0) for ch in OPTION_LETTERS}
                out_row["TeacherAnswer"] = gt
                out_row["TeacherAnswers"] = teacher_answers
                out_row["TeacherConsistency"] = sample_consistencies
                out_row["FusionSource"] = "gt_fallback"
                stats["gt_fallback"] += 1

            wf.write(json.dumps(out_row, ensure_ascii=False) + "\n")

    # 报告
    total = stats["total"]
    print(f"\n{'='*55}")
    print(f"多票融合 + 自一致性过滤 统计")
    print(f"{'='*55}")
    print(f"总样本:       {total}")
    print(f"融合成功:     {stats['fused']} ({stats['fused']/total*100:.1f}%)")
    print(f"GT回退:       {stats['gt_fallback']} ({stats['gt_fallback']/total*100:.1f}%)")
    print(f"\n教师一致性分布:")
    for i, name in enumerate(args.teacher_names):
        h = stats['high_consistency'][i]
        m = stats['mid_consistency'][i]
        l = stats['low_consistency'][i]
        mi = stats['missing'][i]
        print(f"  {name}: 高={h} 中={m} 低={l} 缺失={mi}")

    if stats["entropy_vals"]:
        ent_mean = sum(stats["entropy_vals"]) / len(stats["entropy_vals"])
        print(f"\n融合分布平均熵: {ent_mean:.4f}")
    if stats["consistency_vals"]:
        cons_mean = sum(stats["consistency_vals"]) / len(stats["consistency_vals"])
        print(f"教师平均一致性: {cons_mean:.4f}")

    # GT 一致率
    gt_match = 0
    out_rows = load_jsonl(args.output)
    for orow in out_rows:
        gt_ans = orow.get("Answer", "").strip().upper()
        fused_ans = max(OPTION_LETTERS, key=lambda k: orow.get("TeacherDist", {}).get(k, 0.0))
        if fused_ans == gt_ans:
            gt_match += 1
    print(f"融合标签 vs GT 一致率: {gt_match}/{total} ({gt_match/total*100:.2f}%)")
    print(f"Output: {args.output}")

    # 写报告
    if args.report:
        report = {
            "total": total,
            "fused": stats["fused"],
            "gt_fallback": stats["gt_fallback"],
            "fused_ratio": round(stats["fused"] / total, 4),
            "high_thresh": args.high_thresh,
            "low_thresh": args.low_thresh,
            "downweight_factor": args.downweight_factor,
            "gt_agreement": round(gt_match / total, 4),
            "mean_entropy": round(sum(stats["entropy_vals"]) / max(1, len(stats["entropy_vals"])), 4),
            "mean_consistency": round(sum(stats["consistency_vals"]) / max(1, len(stats["consistency_vals"])), 4),
            "teacher_stats": {
                args.teacher_names[i]: {
                    "high": stats["high_consistency"][i],
                    "mid": stats["mid_consistency"][i],
                    "low": stats["low_consistency"][i],
                    "missing": stats["missing"][i],
                }
                for i in range(len(args.teacher_names))
            },
        }
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
