#!/usr/bin/env python3
"""
多 MCQ 数据集统一加载适配器 (数据集扩展的基础设施)

把不同来源的 MCQ 数据集统一转成 mentalDistill 管线用的 jsonl 格式:
  {"Question": str, "Options": {"A":..,"B":..,...}, "Answer": "A".."E", "Subject": str}

支持选项数 K=4 或 5 (写入 _num_options 字段, 供下游 option-token 映射用)。

支持的数据集:
  - mmlu_med   : MMLU 医学相关子集 (clinical_knowledge, professional_medicine,
                 college_medicine, medical_genetics, anatomy)  4选
  - mmlu_full  : MMLU 全部 57 学科 (跨域验证)  4选
  - medqa      : MedQA USMLE (英文执业医考)  通常4-5选
  - cmexam     : 复用本地 15_fulldata_resplit (中文全科医考)  5选

用法:
    export HF_HUB_ENABLE_HF_TRANSFER=0   # 用官方源直连
    python research/distillability/scripts/load_mcq_dataset.py --dataset mmlu_med --out_dir data_ext/mmlu_med
    python research/distillability/scripts/load_mcq_dataset.py --dataset mmlu_full --out_dir data_ext/mmlu_full
    python research/distillability/scripts/load_mcq_dataset.py --dataset medqa --out_dir data_ext/medqa

输出: <out_dir>/{train,val,test}.jsonl  + meta.json (统计信息)
"""
import argparse
import json
import os
import random

LETTERS = ["A", "B", "C", "D", "E"]

# MMLU 医学相关学科
MMLU_MED_SUBJECTS = [
    "clinical_knowledge", "professional_medicine", "college_medicine",
    "medical_genetics", "anatomy", "college_biology", "nutrition",
]


def write_jsonl(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def to_record(question, choices, answer_idx, subject=""):
    """choices: list[str]; answer_idx: int. 返回统一格式 dict 或 None(非法)."""
    choices = [str(c).strip() for c in choices if str(c).strip()]
    k = len(choices)
    if k < 2 or k > 5:
        return None
    if not (0 <= answer_idx < k):
        return None
    options = {LETTERS[i]: choices[i] for i in range(k)}
    return {
        "Question": str(question).strip(),
        "Options": options,
        "Answer": LETTERS[answer_idx],
        "Subject": subject,
        "_num_options": k,
    }


def load_mmlu(subjects):
    from datasets import load_dataset, get_dataset_config_names
    if subjects == "ALL":
        subjects = get_dataset_config_names("cais/mmlu")
        subjects = [s for s in subjects if s not in ("all", "auxiliary_train")]
    splits = {"train": [], "val": [], "test": []}
    for subj in subjects:
        try:
            d = load_dataset("cais/mmlu", subj)
        except Exception as e:
            print(f"  [skip] {subj}: {e}")
            continue
        # MMLU: test/validation/dev. dev 太小, 合到 train; test->test; validation->val
        for split_src, split_dst in [("dev", "train"), ("validation", "val"), ("test", "test")]:
            if split_src not in d:
                continue
            for ex in d[split_src]:
                rec = to_record(ex["question"], ex["choices"], int(ex["answer"]), subj)
                if rec:
                    splits[split_dst].append(rec)
    return splits


def load_medqa():
    """MedQA USMLE 英文. 尝试常见的 HF repo。"""
    from datasets import load_dataset
    # 常见 repo: GBaker/MedQA-USMLE-4-options (4选, 干净)
    d = load_dataset("GBaker/MedQA-USMLE-4-options")
    splits = {"train": [], "val": [], "test": []}
    split_map = {"train": "train", "validation": "val", "test": "test"}
    for src, dst in split_map.items():
        if src not in d:
            continue
        for ex in d[src]:
            # 字段: question, options(dict A-D), answer_idx 或 answer
            q = ex.get("question", "")
            opts = ex.get("options")  # dict {"A":..,"B":..}
            if isinstance(opts, dict):
                choices = [opts[k] for k in sorted(opts.keys())]
            else:
                choices = list(opts) if opts else []
            ans = ex.get("answer_idx") or ex.get("answer")
            # answer_idx 可能是 "A".."D"
            if isinstance(ans, str) and ans.strip().upper() in LETTERS:
                ai = LETTERS.index(ans.strip().upper())
            else:
                try:
                    ai = int(ans)
                except Exception:
                    continue
            rec = to_record(q, choices, ai, "medqa_usmle")
            if rec:
                splits[dst].append(rec)
    return splits


def load_cmexam_local():
    """复用本地 CMExam, 转成统一格式(加 Subject/_num_options)。"""
    base = "15_fulldata_resplit/data"
    splits = {"train": [], "val": [], "test": []}
    src_map = {"train": "train.jsonl", "val": "val.jsonl", "test": "test.jsonl"}
    for dst, fn in src_map.items():
        p = os.path.join(base, fn)
        if not os.path.exists(p):
            continue
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            opts = ex.get("Options", {})
            ans = str(ex.get("Answer", "")).strip().upper()
            if ans not in LETTERS:
                continue
            present = [L for L in LETTERS if L in opts]
            rec = {
                "Question": ex.get("Question", ""),
                "Options": {L: opts[L] for L in present},
                "Answer": ans,
                "Subject": str(ex.get("Medical Discipline", "")).strip(),
                "_num_options": len(present),
            }
            splits[dst].append(rec)
    return splits


def maybe_resplit(splits, seed=42):
    """若某些数据集缺 train/val(如 MMLU test 占绝大多数), 从 test 切出 train/val。"""
    rng = random.Random(seed)
    # 若 train 太小 (<200) 而 test 很大, 重新切分: 60% train / 15% val / 25% test
    total = sum(len(v) for v in splits.values())
    if len(splits["train"]) < 200 and len(splits["test"]) > 500:
        allrows = splits["train"] + splits["val"] + splits["test"]
        rng.shuffle(allrows)
        n = len(allrows)
        n_tr = int(n * 0.60)
        n_va = int(n * 0.15)
        splits = {
            "train": allrows[:n_tr],
            "val": allrows[n_tr:n_tr + n_va],
            "test": allrows[n_tr + n_va:],
        }
        print(f"  [resplit] 原train太小, 重切为 train={len(splits['train'])} "
              f"val={len(splits['val'])} test={len(splits['test'])}")
    return splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=["mmlu_med", "mmlu_full", "medqa", "cmexam"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"=== 加载 {args.dataset} ===")
    if args.dataset == "mmlu_med":
        splits = load_mmlu(MMLU_MED_SUBJECTS)
    elif args.dataset == "mmlu_full":
        splits = load_mmlu("ALL")
    elif args.dataset == "medqa":
        splits = load_medqa()
    elif args.dataset == "cmexam":
        splits = load_cmexam_local()
    else:
        raise ValueError(f"unknown dataset: {args.dataset}")

    splits = maybe_resplit(splits, args.seed)

    # 写出
    from collections import Counter
    meta = {"dataset": args.dataset, "splits": {}, "num_options_dist": {}, "subjects": {}}
    allrows = []
    for split, rows in splits.items():
        write_jsonl(rows, os.path.join(args.out_dir, f"{split}.jsonl"))
        meta["splits"][split] = len(rows)
        allrows += rows
    ko = Counter(r["_num_options"] for r in allrows)
    subj = Counter(r["Subject"] for r in allrows)
    meta["num_options_dist"] = dict(ko)
    meta["subjects"] = dict(subj.most_common(20))
    meta["total"] = len(allrows)
    json.dump(meta, open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    print(f"  splits: {meta['splits']}")
    print(f"  选项数分布: {meta['num_options_dist']}")
    print(f"  学科数: {len(subj)}")
    print(f"  -> {args.out_dir}")


if __name__ == "__main__":
    main()
