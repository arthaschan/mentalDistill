#!/usr/bin/env python3
"""中文牙科 test_dental(125) 评估：测 base 零样本 或 base+adapter 训练后，与 DeepSeek 老师 79.20% 对比。

用法:
  python eval_cn_dental_test.py                # base 零样本
  python eval_cn_dental_test.py --adapter RUN/Qwen3_cn_a00_s11/best   # 训练后
"""
import argparse
import json
import os
import sys

import torch

CN_DATA = "15_fulldata_resplit/data"
TEST = f"{CN_DATA}/test_dental.jsonl"
TEACHER_ACC = 79.20   # DeepSeek 老师（Module 15 README 口径，test_dental 125 题）

os.environ["DISTILL_USE_CHAT_TEMPLATE"] = "1"
os.environ["DISTILL_PROMPT_LANG"] = "zh"
sys.path.insert(0, "shared")
from train_choice_head_distill import (  # noqa: E402
    apply_prompt_template, build_mcq_prompt, extract_answer_char,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3-32B")
    ap.add_argument("--adapter", default="")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(TEST) if l.strip()]

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map=device,
    )
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    correct = total = 0
    for r in rows:
        q = r.get("Question", "")
        opts = r.get("Options", "")
        gt = str(r.get("Answer", "")).strip().upper()
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

    acc = round(100.0 * correct / total, 2) if total else 0.0
    tag = f"训练后({os.path.basename(args.adapter)})" if args.adapter else "零样本"
    print(f"=== 中文牙科 test_dental({total}题) {tag} ===")
    print(f"学生: {acc}% ({correct}/{total})")
    print(f"老师 DeepSeek: {TEACHER_ACC}%")
    print(f"Δ(学生-老师): {acc - TEACHER_ACC:+.2f}pp  -> {'超越' if acc > TEACHER_ACC else '未超越'}")


if __name__ == "__main__":
    main()
