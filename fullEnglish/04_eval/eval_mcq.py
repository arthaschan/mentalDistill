#!/usr/bin/env python3
"""fullEnglish — 统一英文评估器 (canonical eval, 与训练 prompt 完全一致).

对 base 模型 (可选 + LoRA adapter) 在测试集上做确定性生成, 输出字母, 与 GT 字母比对.
与 train_choice_head_distill.py 的 DISTILL_PROMPT_LANG=en 使用同一套 ChatML prompt,
保证训练/评估口径一致.

支持任意选项数 (MedQA=5 / MedMCQA&MMLU=4 / PubMedQA=3): 选项已内嵌在 Options 字符串,
答案按字母 A-E 比对, 与选项数无关.

Usage:
    python3 fullEnglish/04_eval/eval_mcq.py \
        --base_model models/Qwen2.5-32B-Instruct \
        [--adapter_dir runs/xxx/best] \
        --test_data fullEnglish/00_data/out/test_medqa.jsonl \
        [--label medqa] [--max_n 0] [--wrong_log out/wrong.jsonl]
"""
import argparse
import json
import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LETTERS = ["A", "B", "C", "D", "E"]

# 与 trainer 的 DISTILL_PROMPT_LANG=en 完全一致
SYSTEM_LINE = ("You are a medical expert. Output exactly one letter "
               "(A, B, C, D, or E) as the answer, with no explanation or spaces.\n")


def build_prompt(q, opts):
    user_block = f"Question: {q}\nOptions:\n{opts}\n"
    return (
        "<|im_start|>system\n" + SYSTEM_LINE + "<|im_end|>\n"
        "<|im_start|>user\n" + user_block + "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def extract_answer_char(text):
    for ch in text.strip().upper():
        if ch in LETTERS:
            return ch
    return ""


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter_dir", default=None)
    ap.add_argument("--test_data", required=True)
    ap.add_argument("--label", default="test", help="测试集名, 用于报告")
    ap.add_argument("--max_n", type=int, default=0, help="只评测前 N 题 (0=全部)")
    ap.add_argument("--wrong_log", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"加载 base 模型: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map=device,
        trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if args.adapter_dir:
        from peft import PeftModel
        print(f"加载 LoRA adapter: {args.adapter_dir}")
        model = PeftModel.from_pretrained(model, args.adapter_dir)
    model.eval()

    rows = load_jsonl(args.test_data)
    if args.max_n and 0 < args.max_n < len(rows):
        rows = rows[:args.max_n]

    correct = 0
    total = 0
    wrong = []
    per_src_correct = {}
    per_src_total = {}
    src_field = "source" if any("source" in r for r in rows[:1]) else None

    for item in rows:
        q = item.get("Question", "")
        opts = item.get("Options", "")
        gt = str(item.get("Answer", "")).strip().upper()
        if not q or not opts or gt not in LETTERS:
            continue
        total += 1
        if src_field:
            s = item.get("source", "?")
            per_src_total[s] = per_src_total.get(s, 0) + 1

        prompt = build_prompt(q, opts)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=4, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        gen = tokenizer.decode(out[0][inputs["input_ids"].size(1):], skip_special_tokens=True)
        pred = extract_answer_char(gen)
        if pred == gt:
            correct += 1
            if src_field:
                per_src_correct[item.get("source", "?")] = per_src_correct.get(item.get("source", "?"), 0) + 1
        else:
            wrong.append({"Question": q, "Options": opts, "Answer": gt,
                          "Predicted": pred, "Raw": gen.strip()})

    acc = 100 * correct / total if total else 0.0
    print(f"\n[{args.label}] 准确率: {acc:.2f}% ({correct}/{total})")
    if src_field and per_src_total:
        print("  分源准确率:")
        for s in sorted(per_src_total):
            c = per_src_correct.get(s, 0)
            print(f"    {s:10s} {100*c/per_src_total[s]:.2f}% ({c}/{per_src_total[s]})")

    if args.wrong_log and wrong:
        os.makedirs(os.path.dirname(args.wrong_log), exist_ok=True)
        with open(args.wrong_log, "w", encoding="utf-8") as f:
            for w in wrong:
                f.write(json.dumps(w, ensure_ascii=False) + "\n")
        print(f"  错误样本 -> {args.wrong_log}")

    return acc


if __name__ == "__main__":
    main()
