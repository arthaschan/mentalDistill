#!/usr/bin/env python3
"""评估 PED 手段①：Qwen3-32B 用"差点答对"选题集训练后的 adapter（α=0 × 3 seed）。

- 与 eval_qwen3.py 完全同口径，只把 run 名从 Qwen3_a00_{seed} 换成 Qwen3_ped_ar_{seed}。
- 产出 runs/eval_results_qwen3_ped_ar.json，用于和主线增益 3.75pp（零样本 73.84%→训练后 77.60%）对比。
"""
import json
import os
import statistics
import sys

import torch

FE = "fullEnglish/03_main_distill"
DATA = "fullEnglish/00_data/out"
RUN = f"{FE}/runs"
STUDENT = "models/Qwen3-32B"
SEEDS = ["s11", "s42", "s8"]
TEST_SETS = ["test_medqa", "test_medmcqa", "test_mmlu", "test_pubmedqa"]
SET_COUNTS = {"test_medqa": 1273, "test_medmcqa": 4183, "test_mmlu": 2837, "test_pubmedqa": 1000}

os.environ["DISTILL_USE_CHAT_TEMPLATE"] = "1"
os.environ["DISTILL_PROMPT_LANG"] = "en"
sys.path.insert(0, "shared")
from train_choice_head_distill import (  # noqa: E402
    apply_prompt_template, build_mcq_prompt, extract_answer_char,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_rows(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def eval_set(model, tok, path):
    correct = total = 0
    for item in load_rows(path):
        q = item.get("Question", "")
        opts = item.get("Options", "")
        gt = str(item.get("Answer", "")).strip().upper()
        if not q or not opts or gt not in "ABCDE":
            continue
        total += 1
        sys_line, user_block = build_mcq_prompt(q, opts)
        prompt, _ = apply_prompt_template(tok, sys_line, user_block)
        inputs = tok(prompt, return_tensors="pt", truncation=True).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=16, do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        gen = tok.decode(out[0][inputs["input_ids"].size(1):], skip_special_tokens=True)
        if extract_answer_char(gen) == gt:
            correct += 1
    return round(100.0 * correct / total, 2) if total else 0.0


def teacher_acc():
    teacher = {}
    for s in ["test_medqa", "test_medmcqa", "test_mmlu"]:
        p = f"{FE}/labels/teacher_{s}.jsonl"
        n = c = 0
        for line in open(p):
            r = json.loads(line)
            gt = str(r.get("OriginalAnswer") or r.get("Answer", "")).strip().upper()
            ta = str(r.get("TeacherAnswer") or r.get("Answer", "")).strip().upper()
            if gt in "ABCDE" and ta in "ABCDE":
                n += 1
                c += int(ta == gt)
        teacher[s] = round(100 * c / n, 2) if n else None
    return teacher


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(STUDENT, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    teacher = teacher_acc()
    print(f"教师 DeepSeek: {teacher}", flush=True)
    zeroshot = json.load(open(f"{RUN}/eval_results_qwen3_zeroshot.json"))["zeroshot"]
    # 主线基准（3-seed 均值，α=0 全量训练）
    baseline = json.load(open(f"{RUN}/eval_results_qwen3.json"))

    per_seed = {s: {} for s in SEEDS}
    for seed in SEEDS:
        ad = f"{RUN}/Qwen3_ped_ar_{seed}/best"
        print(f"=== 评估 Qwen3_ped_ar_{seed} ===", flush=True)
        base = AutoModelForCausalLM.from_pretrained(
            STUDENT, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map=device,
        )
        from peft import PeftModel
        model = PeftModel.from_pretrained(base, ad)
        model.eval()
        for t in TEST_SETS:
            per_seed[seed][t] = eval_set(model, tok, f"{DATA}/{t}.jsonl")
            print(f"  {t:14s} {per_seed[seed][t]}%", flush=True)
        del model, base
        torch.cuda.empty_cache()

    summary = {}
    for t in TEST_SETS:
        vals = [per_seed[s][t] for s in SEEDS]
        summary[t] = (round(statistics.mean(vals), 2), round(statistics.stdev(vals), 2))

    mcq = ["test_medqa", "test_medmcqa", "test_mmlu"]
    comb_n = sum(SET_COUNTS[s] for s in mcq)
    comb_t = sum(teacher[s] * SET_COUNTS[s] for s in mcq) / comb_n
    comb_z = sum(zeroshot[s] * SET_COUNTS[s] for s in mcq) / comb_n
    comb_s = sum(summary[s][0] * SET_COUNTS[s] for s in mcq) / comb_n
    # 主线基准的组合口径（eval_results_qwen3.json 里存的 combined_student）
    comb_base = baseline.get("combined_student", 0.0)

    print("\n=== PED① 选题蒸馏结果 (mean±std over 3 seed) ===")
    for t in TEST_SETS:
        m, sd = summary[t]
        t_ = teacher.get(t)
        z = zeroshot[t]
        print(f"  {t:14s} 零样本 {z}%  学生 {m}±{sd}  教师 {t_}")

    print(f"\n组合 MCQ: 教师 {comb_t:.2f}%  零样本 {comb_z:.2f}%  PED①学生 {comb_s:.2f}%")
    print(f"  主线基准学生(全量训练) = {comb_base:.2f}%")
    print(f"  PED①增益(零样本→学生) = {comb_s - comb_z:+.2f}pp  (主线增益 = {comb_base - comb_z:+.2f}pp)")
    print(f"  Δ(PED①-主线) = {comb_s - comb_base:+.2f}pp")
    print(f"  Δ(PED①学生-教师flash) = {comb_s - comb_t:+.2f}pp  -> {'超越' if comb_s > comb_t else '未超越'} flash")

    json.dump({"teacher": teacher, "zeroshot": zeroshot, "per_seed": per_seed,
               "summary": summary, "combined_teacher": round(comb_t, 2),
               "combined_zeroshot": round(comb_z, 2), "combined_student": round(comb_s, 2),
               "combined_baseline": comb_base},
              open(f"{RUN}/eval_results_qwen3_ped_ar.json", "w"), ensure_ascii=False, indent=2)
    print(f"\n-> {RUN}/eval_results_qwen3_ped_ar.json")


if __name__ == "__main__":
    main()
