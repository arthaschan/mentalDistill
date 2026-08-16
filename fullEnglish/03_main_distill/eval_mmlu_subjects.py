#!/usr/bin/env python3
"""MMLU 逐题评估 + 按学科(subject)分组对比教师，找「学生超越老师」的学科分支。

- 评估 32B 学生（α=0 × 3 seed，bf16，Qwen 硬编码 prompt，与主线训练一致）。
- 保存每题预测，按 subject 分组，对比 DeepSeek 教师（teacher_test_mmlu.jsonl）。
- 产出 runs/mmlu_subject_analysis.json + 打印学科分支表。
"""
import json
import os
import statistics
import sys
from collections import defaultdict

import torch

FE = "fullEnglish/03_main_distill"
DATA = "fullEnglish/00_data/out"
RUN = f"{FE}/runs"
STUDENT = "models/Qwen2.5-32B-Instruct"
SEEDS = ["s11", "s42", "s8"]
LETTERS = ["A", "B", "C", "D", "E"]

os.environ["DISTILL_PROMPT_LANG"] = "en"  # 主线训练用英文 prompt
# 不设 DISTILL_USE_CHAT_TEMPLATE -> 用 Qwen 硬编码 <|im_start|>（与 32B 主线一致）
sys.path.insert(0, "shared")
from train_choice_head_distill import (  # noqa: E402
    apply_prompt_template, build_mcq_prompt, extract_answer_char,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_rows(path):
    rows = []
    for line in open(path):
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        STUDENT, eos_token="<|endoftext|>", pad_token="<|endoftext|>",
        unk_token="<|endoftext|>", trust_remote_code=True,
    )
    rows = load_rows(f"{DATA}/test_mmlu.jsonl")

    # 教师标签 uid -> TeacherAnswer
    teacher = {}
    for r in load_rows(f"{FE}/labels/teacher_test_mmlu.jsonl"):
        if r.get("uid"):
            ta = str(r.get("TeacherAnswer") or r.get("Answer", "")).strip().upper()
            gt = str(r.get("OriginalAnswer") or r.get("Answer", "")).strip().upper()
            teacher[r["uid"]] = (ta, gt)

    # 学生 3 seed 逐题预测
    per_seed_pred = {s: {} for s in SEEDS}
    for seed in SEEDS:
        ad = f"{RUN}/32B_a00_{seed}/best"
        print(f"=== 评估 32B_a00_{seed} 于 MMLU ===", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            STUDENT, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map=device,
        )
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, ad)
        model.eval()
        for i, r in enumerate(rows):
            q = r.get("Question", "")
            opts = r.get("Options", "")
            uid = r.get("uid", str(i))
            sys_line, user_block = build_mcq_prompt(q, opts)
            prompt, _ = apply_prompt_template(tokenizer, sys_line, user_block)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=4, do_sample=False,
                                     pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
            gen = tokenizer.decode(out[0][inputs["input_ids"].size(1):], skip_special_tokens=True)
            per_seed_pred[seed][uid] = extract_answer_char(gen)
        del model
        torch.cuda.empty_cache()

    # 按 subject 分组对比
    subj_rows = defaultdict(list)
    for r in rows:
        subj_rows[r.get("subject", "(无)")].append(r)

    out = {}
    print("\n=== MMLU 学科分支：教师 vs 32B 学生(3 seed mean) ===")
    print(f"{'subject':28s} {'n':>5s} {'教师':>7s} {'学生':>8s} {'Δ(学生-教师)':>12s}")
    for subj, subrows in sorted(subj_rows.items(), key=lambda kv: -len(kv[1])):
        t_c = t_n = 0
        s_accs = []
        for r in subrows:
            uid = r.get("uid")
            if uid not in teacher:
                continue
            ta, gt = teacher[uid]
            if ta in LETTERS and gt in LETTERS:
                t_n += 1
                t_c += int(ta == gt)
        t_acc = round(100 * t_c / t_n, 2) if t_n else None
        for s in SEEDS:
            c = sum(1 for r in subrows if (r.get("uid") in teacher and
                                           per_seed_pred[s].get(r.get("uid")) == str(r.get("Answer", "")).strip().upper()))
            s_accs.append(round(100 * c / t_n, 2) if t_n else 0.0)
        s_mean = round(statistics.mean(s_accs), 2)
        d = round(s_mean - t_acc, 2) if t_acc is not None else None
        flag = "  <<< 超越" if (d is not None and d > 0) else ""
        print(f"{subj:28s} {t_n:5d} {t_acc:7.2f} {s_mean:8.2f} {d:>+12.2f}{flag}")
        out[subj] = {"n": t_n, "teacher_acc": t_acc, "student_mean": s_mean,
                     "delta": d, "per_seed": s_accs}

    json.dump(out, open(f"{RUN}/mmlu_subject_analysis.json", "w"), ensure_ascii=False, indent=2)
    print(f"\n-> {RUN}/mmlu_subject_analysis.json")


if __name__ == "__main__":
    main()
