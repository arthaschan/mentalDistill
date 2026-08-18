#!/usr/bin/env python3
"""中文牙科单模型评估（零样本 或 指定 adapter），test_dental 125 题。

用法:
  python eval_single.py                 # 零样本
  python eval_single.py --adapter runs/Qwen3_cn_a00_s11/best
"""
import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
TEST = os.path.join(DATA, "test_dental.jsonl")
STUDENT = os.path.join(ROOT, "..", "models", "Qwen3-32B")
TEACHER_ACC = 79.20

os.environ["DISTILL_USE_CHAT_TEMPLATE"] = "1"
os.environ["DISTILL_PROMPT_LANG"] = "zh"
sys.path.insert(0, os.path.join(ROOT, "..", "shared"))
from train_choice_head_distill import (  # noqa: E402
    apply_prompt_template, build_mcq_prompt, extract_answer_char,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(TEST) if l.strip()]
    tok = AutoTokenizer.from_pretrained(STUDENT, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        STUDENT, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map=device,
    )
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, os.path.join(ROOT, args.adapter))
    model.eval()

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

    acc = round(100.0 * c / len(rows), 2)
    tag = f"训练后({args.adapter})" if args.adapter else "零样本"
    print(f"=== 中文牙科 test_dental({len(rows)}题) {tag} ===")
    print(f"学生 {acc}%  老师 {TEACHER_ACC}%  Δ {acc - TEACHER_ACC:+.2f}pp  "
          f"{'超越' if acc > TEACHER_ACC else '未超越'}")


if __name__ == "__main__":
    main()
