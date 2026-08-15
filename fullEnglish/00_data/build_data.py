#!/usr/bin/env python3
"""Phase 0 — fullEnglish 数据装配 (zero GPU, runs anywhere).

把 fullEnglish/data/*.jsonl 的「统一格式」转成 trainer/eval/label 三套脚本共用的
「trainer 格式」，并切分 train / val / test / screening 子集，最后输出现实审计。

统一格式字段 (README):
    id, source, split, subject, question, context, options(list), answer_idx(int), answer(str)

trainer 格式 (与 15_fulldata_resplit / english 完全一致):
    uid, Question, Options("A. xxx\nB. xxx\n..."), Answer(大写字母 A-E),
    n_options, source, subject, answer_idx, answer_text

切分决策 (与中文 Module 15 同思路, 保证「学生 vs 老师」在同题集可比):
    train           : MedQA train (10178) + MedMCQA train(抽样 TRAIN_MEDMCQA_SAMPLE)
                      + MMLU 12 科目 validation (~300)
    val (选点)      : MedQA dev (1272)         —— 干净 hold-out, 不进 train
    test (held-out) : test_medqa (1273) / test_medmcqa (4183) / test_mmlu (~2837)
    test_pubmedqa   : 3 选 1 判断题, 蒸馏全程不用, 仅作「泛化到判断题」额外评测
    screen_input    : 每个 test 集抽样 200 题 (seed=2026), 用于教师预评估 + 学生零样本地板

运行:
    python3 fullEnglish/00_data/build_data.py [--medmcqa_sample N] [--screen_per_source N]
"""
import argparse
import json
import os
import random
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # fullEnglish/
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "00_data", "out")
REP = os.path.join(ROOT, "00_data", "reports")

# MMLU 12 医学科目 (README 三)
MMLU_SUBJECTS = [
    "anatomy", "clinical_knowledge", "college_biology", "college_medicine",
    "medical_genetics", "professional_medicine", "high_school_biology",
    "nutrition", "virology", "human_aging", "human_sexuality",
    "professional_psychology",
]

LETTERS = "ABCDE"


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def to_trainer(r):
    """统一格式 -> trainer 格式。options(list) -> "A. text\\nB. ...", answer_idx -> 字母。"""
    opts = r.get("options") or []
    letters = LETTERS[:len(opts)]
    opt_str = "\n".join(f"{L}. {str(o).strip()}" for L, o in zip(letters, opts))
    idx = int(r.get("answer_idx", 0))
    ans_letter = letters[idx] if 0 <= idx < len(letters) else ""
    return {
        "uid": r.get("id", ""),
        "Question": str(r.get("question", "")).strip(),
        "Options": opt_str,
        "Answer": ans_letter,
        "n_options": len(opts),
        "source": r.get("source", ""),
        "subject": r.get("subject", ""),
        "answer_idx": idx,
        "answer_text": str(r.get("answer", "")).strip(),
        "context": str(r.get("context", "")).strip(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--medmcqa_sample", type=int, default=10000,
                    help="MedMCQA train 抽样量 (默认 10000; 0=全部 182822)")
    ap.add_argument("--screen_per_source", type=int, default=200,
                    help="每个 test 集抽多少题做教师预评估")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    os.makedirs(REP, exist_ok=True)
    rng = random.Random(2026)

    # ---------- 读取并转换所有 MCQ 源 ----------
    medqa_train = [to_trainer(r) for r in load_jsonl(os.path.join(DATA, "medqa_train.jsonl"))]
    medqa_dev = [to_trainer(r) for r in load_jsonl(os.path.join(DATA, "medqa_dev.jsonl"))]
    medqa_test = [to_trainer(r) for r in load_jsonl(os.path.join(DATA, "medqa_test.jsonl"))]
    medmcqa_train = [to_trainer(r) for r in load_jsonl(os.path.join(DATA, "medmcqa_train.jsonl"))]
    medmcqa_val = [to_trainer(r) for r in load_jsonl(os.path.join(DATA, "medmcqa_validation.jsonl"))]

    mmlu_val, mmlu_test = [], []
    for subj in MMLU_SUBJECTS:
        vp = os.path.join(DATA, f"mmlu_{subj}_validation.jsonl")
        tp = os.path.join(DATA, f"mmlu_{subj}_test.jsonl")
        if os.path.exists(vp):
            mmlu_val += [to_trainer(r) for r in load_jsonl(vp)]
        if os.path.exists(tp):
            mmlu_test += [to_trainer(r) for r in load_jsonl(tp)]

    pubmedqa = [to_trainer(r) for r in load_jsonl(os.path.join(DATA, "pubmedqa_labeled.jsonl"))]

    # ---------- MedMCQA train 抽样 ----------
    if args.medmcqa_sample > 0 and len(medmcqa_train) > args.medmcqa_sample:
        medmcqa_train = rng.sample(medmcqa_train, args.medmcqa_sample)

    train = medqa_train + medmcqa_train + mmlu_val
    val = medqa_dev

    # ---------- 写训练/验证/测试 ----------
    write_jsonl(os.path.join(OUT, "train.jsonl"), train)
    write_jsonl(os.path.join(OUT, "val.jsonl"), val)
    write_jsonl(os.path.join(OUT, "test_medqa.jsonl"), medqa_test)
    write_jsonl(os.path.join(OUT, "test_medmcqa.jsonl"), medmcqa_val)
    write_jsonl(os.path.join(OUT, "test_mmlu.jsonl"), mmlu_test)
    write_jsonl(os.path.join(OUT, "test_pubmedqa.jsonl"), pubmedqa)

    # ---------- screening 子集 (教师预评估 + 学生零样本地板) ----------
    screen = []
    for name, rows in [("medqa", medqa_test), ("medmcqa", medmcqa_val), ("mmlu", mmlu_test)]:
        k = min(args.screen_per_source, len(rows))
        screen += rng.sample(rows, k)
    rng.shuffle(screen)
    write_jsonl(os.path.join(OUT, "screen_input.jsonl"), screen)

    # ---------- 审计 ----------
    def opt_hist(rows):
        return dict(sorted(Counter(r["n_options"] for r in rows).items()))

    def src_hist(rows):
        return dict(sorted(Counter(r["source"] for r in rows).items()))

    def ans_bal(rows):
        c = Counter(r["Answer"] for r in rows)
        n = len(rows)
        return {k: round(100 * v / n, 1) for k, v in sorted(c.items())}

    report = {
        "medmcqa_train_sample": args.medmcqa_sample,
        "screen_per_source": args.screen_per_source,
        "train": {"n": len(train), "source": src_hist(train), "n_options": opt_hist(train)},
        "val": {"n": len(val), "source": src_hist(val), "n_options": opt_hist(val)},
        "test_medqa": {"n": len(medqa_test), "n_options": opt_hist(medqa_test)},
        "test_medmcqa": {"n": len(medmcqa_val), "n_options": opt_hist(medmcqa_val)},
        "test_mmlu": {"n": len(mmlu_test), "n_options": opt_hist(mmlu_test),
                      "subjects": dict(sorted(Counter(r["subject"] for r in mmlu_test).items()))},
        "test_pubmedqa": {"n": len(pubmedqa), "n_options": opt_hist(pubmedqa),
                          "answer_balance": ans_bal(pubmedqa)},
        "screen_input": {"n": len(screen), "source": src_hist(screen)},
    }
    with open(os.path.join(REP, "data_audit.json"), "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 打印简洁汇总
    print("=" * 64)
    print("fullEnglish Phase 0 — 数据装配结果")
    print("=" * 64)
    print(f"train           : {len(train):>7}   (medqa {len(medqa_train)} + medmcqa {len(medmcqa_train)} + mmlu_val {len(mmlu_val)})")
    print(f"val (选点)      : {len(val):>7}   (medqa_dev)")
    print(f"test_medqa      : {len(medqa_test):>7}   (5 选 1)")
    print(f"test_medmcqa    : {len(medmcqa_val):>7}   (4 选 1)")
    print(f"test_mmlu       : {len(mmlu_test):>7}   (4 选 1, {len(MMLU_SUBJECTS)} 科目)")
    print(f"test_pubmedqa   : {len(pubmedqa):>7}   (3 选 1 判断题, 泛化评测)")
    print(f"screen_input    : {len(screen):>7}   (教师预评估子集)")
    print(f"n_options 分布  : train={opt_hist(train)}")
    print(f"train 来源      : {src_hist(train)}")
    print(f"-> 输出目录: {OUT}")
    print(f"-> 审计报告: {REP}/data_audit.json")


if __name__ == "__main__":
    main()
