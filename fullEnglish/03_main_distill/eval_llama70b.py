#!/usr/bin/env python3
"""评估 Llama-3.3-70B 学生（α=0 × 3 seed, QLoRA）在 4 个测试集上的准确率，
并与 DeepSeek 教师同集对比；同时评估基础模型零样本（补精确 headroom / 蒸馏增益）。

- 4bit QLoRA 加载（70B bf16 装不下 95GB）。
- 用 tokenizer 自带 chat template（与训练一致）。
- 产出 runs/eval_results_llama70b.json。
"""
import json
import os
import statistics
import sys

import torch

FE = "fullEnglish/03_main_distill"
DATA = "fullEnglish/00_data/out"
RUN = f"{FE}/runs"
STUDENT = "models/Llama-3.3-70B-Instruct"
SEEDS = ["s11", "s42", "s8"]
TEST_SETS = ["test_medqa", "test_medmcqa", "test_mmlu", "test_pubmedqa"]
SET_COUNTS = {"test_medqa": 1273, "test_medmcqa": 4183, "test_mmlu": 2837, "test_pubmedqa": 1000}

os.environ["DISTILL_USE_CHAT_TEMPLATE"] = "1"
os.environ["DISTILL_PROMPT_LANG"] = "en"
sys.path.insert(0, "shared")
from train_choice_head_distill import (  # noqa: E402
    apply_prompt_template, build_mcq_prompt, extract_answer_char, load_base_model,
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
            out = model.generate(**inputs, max_new_tokens=4, do_sample=False,
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
    tok = AutoTokenizer_proxy()
    teacher = teacher_acc()
    print(f"教师 DeepSeek: {teacher}", flush=True)

    # ① 基础模型零样本（补 headroom / 增益）
    print("=== ① Llama-70B 基础模型零样本 ===", flush=True)
    base = load_base_model(STUDENT, "4bit", device)
    base.eval()
    zeroshot = {}
    for t in TEST_SETS:
        zeroshot[t] = eval_set(base, tok, f"{DATA}/{t}.jsonl")
        print(f"  {t:14s} {zeroshot[t]}%", flush=True)
    del base
    torch.cuda.empty_cache()

    # ② 训练后 3 seed
    per_seed = {s: {} for s in SEEDS}
    for seed in SEEDS:
        ad = f"{RUN}/Llama70B_a00_{seed}/best"
        print(f"=== ② 评估 Llama70B_a00_{seed} ===", flush=True)
        base = load_base_model(STUDENT, "4bit", device)
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

    # 组合 MCQ 加权
    mcq = ["test_medqa", "test_medmcqa", "test_mmlu"]
    comb_n = sum(SET_COUNTS[s] for s in mcq)
    comb_t = sum(teacher[s] * SET_COUNTS[s] for s in mcq) / comb_n
    comb_z = sum(zeroshot[s] * SET_COUNTS[s] for s in mcq) / comb_n
    comb_s = sum(summary[s][0] * SET_COUNTS[s] for s in mcq) / comb_n

    print("\n=== Llama-70B 学生 α=0 结果 (mean±std over 3 seed) ===")
    for t in TEST_SETS:
        m, sd = summary[t]
        t_ = teacher.get(t)
        z = zeroshot[t]
        d_t = round(m - t_, 2) if t_ is not None else None
        print(f"  {t:14s} 零样本 {z}%  学生 {m}±{sd}  教师 {t_}  Δ(学生-教师) {d_t}")

    print(f"\n组合 MCQ: 教师 {comb_t:.2f}%  零样本 {comb_z:.2f}%  学生 {comb_s:.2f}%")
    print(f"  Δ(学生-教师) = {comb_s - comb_t:+.2f}pp  -> {'超越' if comb_s > comb_t else '未超越'}")
    print(f"  蒸馏增益(零样本→学生) = {comb_s - comb_z:+.2f}pp")

    json.dump({"teacher": teacher, "zeroshot": zeroshot, "per_seed": per_seed,
               "summary": summary, "combined_teacher": round(comb_t, 2),
               "combined_zeroshot": round(comb_z, 2), "combined_student": round(comb_s, 2)},
              open(f"{RUN}/eval_results_llama70b.json", "w"), ensure_ascii=False, indent=2)
    print(f"\n-> {RUN}/eval_results_llama70b.json")


def AutoTokenizer_proxy():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(STUDENT, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


if __name__ == "__main__":
    main()
