#!/usr/bin/env python3
"""Llama-70B 无印度零样本评测（不加 adapter）：全科 4110 + 牙科 501。

用途：补 Llama-70B 无印度零样本基线（算 headroom 和增益用）。
单进程只加载一次 Llama-70B(4bit)，避免与老师模型同时驻留导致 OOM。
"""
import json
import os
import sys

os.environ["DISTILL_USE_CHAT_TEMPLATE"] = "1"
os.environ["DISTILL_PROMPT_LANG"] = "en"
sys.path.insert(0, "../shared")

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from train_choice_head_distill import (  # noqa: E402
    apply_prompt_template, build_mcq_prompt, extract_answer_char, load_base_model,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
STU = "../models/Llama-3.3-70B-Instruct"
FULL = "data/test_no_india.jsonl"
DENTAL = "../28_english_dental_noindia/data/test_no_india_dental.jsonl"


def eval_file(model, tok, path):
    samples = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("Question") and r.get("Options") and r.get("Answer"):
            samples.append((r["Question"], r["Options"], str(r["Answer"]).strip().upper()))
    correct = 0
    model.eval()
    with torch.no_grad():
        for q, opts, ans in samples:
            sys_line, user_block = build_mcq_prompt(q, opts)
            prompt, _ = apply_prompt_template(tok, sys_line, user_block)
            inputs = tok(prompt, return_tensors="pt", truncation=True).to(device)
            out = model.generate(**inputs, max_new_tokens=4, do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
            gen = tok.decode(out[0][inputs["input_ids"].size(1):], skip_special_tokens=True)
            if extract_answer_char(gen) == ans:
                correct += 1
    return round(100.0 * correct / len(samples), 2) if samples else None


def main():
    print("[Llama-70B] 加载 4bit 零样本 ...", flush=True)
    tok = AutoTokenizer.from_pretrained(STU, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = load_base_model(STU, "4bit", device)
    base.eval()

    f = eval_file(base, tok, FULL)
    print(f"  全科零样本(4110): {f}%", flush=True)
    d = eval_file(base, tok, DENTAL)
    print(f"  牙科零样本(501): {d}%", flush=True)

    print(f"\n=== Llama-70B 无印度零样本：全科 {f}% / 牙科 {d}% ===", flush=True)


if __name__ == "__main__":
    main()
