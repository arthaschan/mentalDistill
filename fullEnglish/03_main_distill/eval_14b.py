#!/usr/bin/env python3
"""评估 14B 对照 (3 seed × 4 测试集) -> mean±std, 与 DeepSeek 教师同集对比."""
import json
import os
import statistics

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LETTERS = ["A", "B", "C", "D", "E"]
SYSTEM_LINE = ("You are a medical expert. Output exactly one letter "
               "(A, B, C, D, or E) as the answer, with no explanation or spaces.\n")
TEST_SETS = ["test_medqa", "test_medmcqa", "test_mmlu", "test_pubmedqa"]
SET_COUNTS = {"test_medqa": 1273, "test_medmcqa": 4183, "test_mmlu": 2837, "test_pubmedqa": 1000}
SEEDS = ["s11", "s42", "s8"]

FE = "fullEnglish/03_main_distill"
DATA = "fullEnglish/00_data/out"
RUN = f"{FE}/runs"
STUDENT = os.environ.get("BASE_MODEL_14B", "models/Qwen2.5-14B-Instruct")
device = "cuda" if torch.cuda.is_available() else "cpu"


def build_prompt(q, opts):
    return ("<|im_start|>system\n" + SYSTEM_LINE + "<|im_end|>\n"
            "<|im_start|>user\n" + f"Question: {q}\nOptions:\n{opts}\n" + "<|im_end|>\n"
            "<|im_start|>assistant\n")


def extract(text):
    for ch in text.strip().upper():
        if ch in LETTERS:
            return ch
    return ""


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


def eval_set(model, tok, path):
    correct = total = 0
    for item in load_rows(path):
        q, opts, gt = item.get("Question", ""), item.get("Options", ""), str(item.get("Answer", "")).strip().upper()
        if not q or not opts or gt not in LETTERS:
            continue
        total += 1
        inputs = tok(build_prompt(q, opts), return_tensors="pt", truncation=True).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=4, do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        gen = tok.decode(out[0][inputs["input_ids"].size(1):], skip_special_tokens=True)
        if extract(gen) == gt:
            correct += 1
    return round(100.0 * correct / total, 2) if total else 0.0


# 教师同集 (DeepSeek 标签文件)
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

# 评估 14B 3 seed
results = {s: {} for s in SEEDS}
for seed in SEEDS:
    ad = f"{RUN}/14B_a00_{seed}/best"
    print(f"=== 评估 14B_a00_{seed} ===", flush=True)
    model = AutoModelForCausalLM.from_pretrained(STUDENT, torch_dtype=torch.bfloat16,
                                                 device_map=device, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(STUDENT, trust_remote_code=True)
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, ad)
    model.eval()
    for t in TEST_SETS:
        acc = eval_set(model, tok, f"{DATA}/{t}.jsonl")
        results[seed][t] = acc
        print(f"  {t:14s} {acc}%", flush=True)
    del model
    torch.cuda.empty_cache()

# mean±std
print("\n=== 14B 学生 α=0 结果 (mean±std over 3 seed) ===")
summary = {}
for t in TEST_SETS:
    vals = [results[s][t] for s in SEEDS]
    m, sd = statistics.mean(vals), statistics.stdev(vals)
    summary[t] = (round(m, 2), round(sd, 2))
    t_ = teacher.get(t)
    delta = round(m - t_, 2) if t_ is not None else None
    print(f"  {t:14s} 学生 {m}±{sd}  教师 {t_}  Δ {delta}")

# 组合 MCQ 加权
comb_t = sum(teacher[s] * SET_COUNTS[s] for s in ["test_medqa", "test_medmcqa", "test_mmlu"])
comb_s = sum(summary[s][0] * SET_COUNTS[s] for s in ["test_medqa", "test_medmcqa", "test_mmlu"])
comb_n = sum(SET_COUNTS[s] for s in ["test_medqa", "test_medmcqa", "test_mmlu"])
ct, cs = comb_t / comb_n, comb_s / comb_n
print(f"\n组合 MCQ: 教师 {ct:.2f}% vs 学生 {cs:.2f}%  Δ = {cs-ct:+.2f}pp")
print("超越" if cs > ct else "未超越")

json.dump({"teacher": teacher, "per_seed": results, "summary": summary,
           "combined_teacher": round(ct, 2), "combined_student": round(cs, 2)},
          open(f"{RUN}/eval_results_14b.json", "w"), ensure_ascii=False, indent=2)
print(f"\n-> {RUN}/eval_results_14b.json")
