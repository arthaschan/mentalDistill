#!/usr/bin/env python3
"""fullEnglish — 构造 Choice-Head 蒸馏训练文件 (AIEA 配方).

把教师标签映射到 train.jsonl 上, 产出 train_head_distill.jsonl:
  {uid, Question, Options, Answer(=GT 字母), TeacherDist, TeacherAnswer, ...}

教师标签两种来源自动识别:
  - API 硬标签 (labels 含 TeacherAnswer, 无 TeacherDist): hard -> soft (smooth_eps=0.25)
  - 本地真实 logprobs (labels 含 TeacherDist): 直接采用, 不做平滑

Usage:
    python3 fullEnglish/03_main_distill/build_train_head.py \
        --train fullEnglish/00_data/out/train.jsonl \
        --teacher fullEnglish/03_main_distill/labels/teacher_train.jsonl \
        --output fullEnglish/03_main_distill/data/train_head_distill.jsonl \
        [--smooth_eps 0.25]
"""
import argparse
import json
import os

LETTERS = ["A", "B", "C", "D", "E"]


def load_jsonl(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def hard_to_soft(ans, n_opts, smooth_eps=0.25):
    letters = LETTERS[:n_opts] if n_opts in (3, 4, 5) else LETTERS
    d = {}
    for k in letters:
        d[k] = (1.0 - smooth_eps + smooth_eps / len(letters)) if k == ans else (smooth_eps / len(letters))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--smooth_eps", type=float, default=0.25)
    args = ap.parse_args()

    train = load_jsonl(args.train)
    labels = load_jsonl(args.teacher)

    lab_map = {}
    for r in labels:
        uid = r.get("uid")
        ta = str(r.get("TeacherAnswer", "")).strip().upper()
        dist = r.get("TeacherDist")
        if uid and ta in LETTERS:
            lab_map[uid] = {"TeacherAnswer": ta, "TeacherDist": dist}

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out = []
    miss = 0
    teacher_correct = 0
    teacher_total = 0
    for r in train:
        uid = r.get("uid")
        gt = str(r.get("Answer", "")).strip().upper()
        n_opts = int(r.get("n_options", 5))
        lab = lab_map.get(uid)
        if lab is None:
            miss += 1
            ta = gt  # fallback: 无教师标签 -> 用 GT (极少数)
            dist = hard_to_soft(ta, n_opts, args.smooth_eps)
        else:
            ta = lab["TeacherAnswer"]
            if isinstance(lab["TeacherDist"], dict) and lab["TeacherDist"]:
                dist = {k: float(lab["TeacherDist"].get(k, 0.0)) for k in LETTERS}
                s = sum(max(0.0, v) for v in dist.values())
                if s <= 0:
                    dist = hard_to_soft(ta, n_opts, args.smooth_eps)
            else:
                dist = hard_to_soft(ta, n_opts, args.smooth_eps)
            teacher_total += 1
            if ta == gt:
                teacher_correct += 1

        row = dict(r)
        row["TeacherAnswer"] = ta
        row["TeacherDist"] = dist
        out.append(row)

    with open(args.output, "w", encoding="utf-8") as wf:
        for r in out:
            wf.write(json.dumps(r, ensure_ascii=False) + "\n")

    acc = 100 * teacher_correct / teacher_total if teacher_total else 0.0
    print(f"train_head_distill: {len(out)} 条 (教师标签缺失 fallback={miss})")
    print(f"教师训练集准确率: {acc:.2f}%  (教师-GT 不一致 {teacher_total - teacher_correct}/{teacher_total})")
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
