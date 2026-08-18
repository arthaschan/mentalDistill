#!/usr/bin/env python3
"""中文牙科 Qwen3-32B 学生 3-seed 评估（test_dental 125 题），与 DeepSeek 老师 79.20% 对比。

同时测零样本（base）与训练后（3 个 adapter），输出增益与超越幅度。
"""
import json
import os
import statistics
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                    # 22_chinese_dental_surpass/
DATA = os.path.join(ROOT, "data")
RUN = os.path.join(ROOT, "runs")
TEST = os.path.join(DATA, "test_dental.jsonl")
STUDENT = os.path.join(ROOT, "..", "models", "Qwen3-32B")
SEEDS = ["s11", "s42", "s8"]
TEACHER_ACC = 79.20

os.environ["DISTILL_USE_CHAT_TEMPLATE"] = "1"
os.environ["DISTILL_PROMPT_LANG"] = "zh"
sys.path.insert(0, os.path.join(ROOT, "..", "shared"))
from train_choice_head_distill import (  # noqa: E402
    apply_prompt_template, build_mcq_prompt, extract_answer_char,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run_eval(model, tok, rows):
    c = 0
    for r in rows:
        q, opts = r.get("Question", ""), r.get("Options", "")
        gt = str(r.get("Answer", "")).strip().upper()
        if not q or not opts or gt not in "ABCDE":
            continue
        sys_line, user_block = build_mcq_prompt(q, opts)
        prompt, _ = apply_prompt_template(tok, sys_line, user_block)
        inputs = tok(prompt, return_tensors="pt", truncation=True).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=16, do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        gen = tok.decode(out[0][inputs["input_ids"].size(1):], skip_special_tokens=True)
        if extract_answer_char(gen) == gt:
            c += 1
    return round(100.0 * c / len(rows), 2)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    rows = [json.loads(l) for l in open(TEST) if l.strip()]
    tok = AutoTokenizer.from_pretrained(STUDENT, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        STUDENT, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map=device,
    )

    zero = run_eval(base, tok, rows)
    print(f"零样本: {zero}%", flush=True)

    per = {}
    for s in SEEDS:
        ad = os.path.join(RUN, f"Qwen3_cn_a00_{s}", "best")
        model = PeftModel.from_pretrained(base, ad)
        per[s] = run_eval(model, tok, rows)
        print(f"  {s}: {per[s]}%", flush=True)

    m = statistics.mean(per.values())
    sd = statistics.stdev(per.values())
    print(f"\n=== Qwen3-32B 中文牙科 test_dental({len(rows)}题) 3-seed ===")
    print(f"零样本 {zero}%  训练后 {m:.2f}±{sd:.2f}%  ({per})")
    print(f"老师 DeepSeek: {TEACHER_ACC}%")
    print(f"增益(零样本→训练后): {m - zero:+.2f}pp")
    print(f"Δ(学生-老师): {m - TEACHER_ACC:+.2f}pp  -> {'超越' if m > TEACHER_ACC else '未超越'}")

    json.dump({"teacher": TEACHER_ACC, "zeroshot": zero, "per_seed": per,
               "mean": round(m, 2), "std": round(sd, 2),
               "delta": round(m - TEACHER_ACC, 2), "gain": round(m - zero, 2)},
              open(os.path.join(DATA, "eval_results_qwen3_cn_dental.json"), "w"),
              ensure_ascii=False, indent=2)
    print(f"-> {DATA}/eval_results_qwen3_cn_dental.json")


if __name__ == "__main__":
    main()
