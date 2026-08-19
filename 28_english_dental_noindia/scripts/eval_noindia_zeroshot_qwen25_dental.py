#!/usr/bin/env python3
"""Qwen2.5-32B 无印度牙科零样本（不加 adapter，bf16，单进程 501 题）。

用途：补 Qwen2.5-32B 牙科无印度零样本基线（算 headroom/增益，对齐 Llama-70B 牙科）。
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
STU = "../models/Qwen2.5-32B-Instruct"
DENTAL = "data/test_no_india_dental.jsonl"


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
    print("[Qwen2.5-32B] 加载 bf16 零样本 ...", flush=True)
    tok = AutoTokenizer.from_pretrained(STU, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = load_base_model(STU, "none", device).to(device)
    base.eval()

    d = eval_file(base, tok, DENTAL)
    print(f"  牙科零样本(501): {d}%", flush=True)
    print(f"\n=== Qwen2.5-32B 无印度牙科零样本: {d}% ===", flush=True)


if __name__ == "__main__":
    main()
