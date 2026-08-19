#!/usr/bin/env python3
"""27 英文全科·无印度评估：学生(Qwen2.5-32B+adapter) vs 弱教师(Qwen3-32B 零样本)。

口径：data/test_no_india.jsonl（4110 题，medqa+mmlu，无印度）。
"""
import json
import os
import sys

os.environ["DISTILL_USE_CHAT_TEMPLATE"] = "1"
os.environ["DISTILL_PROMPT_LANG"] = "en"
sys.path.insert(0, "../shared")

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from peft import PeftModel  # noqa: E402
from train_choice_head_distill import (  # noqa: E402
    apply_prompt_template, build_mcq_prompt, extract_answer_char, load_base_model,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ADAPTER = "runs/32B_noindia_a00_s42"
TEST = "data/test_no_india.jsonl"
STU = "../models/Qwen2.5-32B-Instruct"
TEA = "../models/Qwen3-32B"


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
    # 学生
    print("[学生] 加载 Qwen2.5-32B + adapter ...", flush=True)
    tok = AutoTokenizer.from_pretrained(STU, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = load_base_model(STU, "none", device).to(device)
    stu = PeftModel.from_pretrained(base, ADAPTER).eval()
    s = eval_file(stu, tok, TEST)
    print(f"  学生(训练后): {s}%", flush=True)
    del stu, base
    torch.cuda.empty_cache()

    # 老师
    print("[老师] 加载 Qwen3-32B 零样本 ...", flush=True)
    tok2 = AutoTokenizer.from_pretrained(TEA, trust_remote_code=True)
    if tok2.pad_token is None:
        tok2.pad_token = tok2.eos_token
    tea = load_base_model(TEA, "none", device).to(device).eval()
    t = eval_file(tea, tok2, TEST)
    print(f"  老师(零样本): {t}%", flush=True)

    print("\n=== 英文全科·无印度：学生 vs 弱教师 ===", flush=True)
    d = round(s - t, 2)
    print(f"  学生 {s}% vs 老师 {t}%  Δ{d:+}  {'超' if s > t else '不超'}", flush=True)


if __name__ == "__main__":
    main()
