#!/usr/bin/env python3
"""Qwen3-32B 零样本评估（4 个测试集），测其 headroom vs DeepSeek。

- bf16 加载，Qwen 硬编码 <|im_start|> prompt（与 Qwen2.5 主线一致）。
- 产出 runs/eval_results_qwen3_zeroshot.json。
"""
import json
import os
import sys

import torch

FE = "fullEnglish/03_main_distill"
DATA = "fullEnglish/00_data/out"
RUN = f"{FE}/runs"
STUDENT = "models/Qwen3-32B"
TEST_SETS = ["test_medqa", "test_medmcqa", "test_mmlu", "test_pubmedqa"]
SET_COUNTS = {"test_medqa": 1273, "test_medmcqa": 4183, "test_mmlu": 2837, "test_pubmedqa": 1000}

os.environ["DISTILL_PROMPT_LANG"] = "en"
sys.path.insert(0, "shared")
from train_choice_head_distill import (  # noqa: E402
    apply_prompt_template, build_mcq_prompt, extract_answer_char,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        STUDENT, eos_token="<|endoftext|>", pad_token="<|endoftext|>",
        unk_token="<|endoftext|>", trust_remote_code=True,
    )
    print(f"加载 Qwen3-32B (bf16) ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        STUDENT, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map=device,
    )
    model.eval()

    zeroshot = {}
    for t in TEST_SETS:
        zeroshot[t] = eval_set(model, tokenizer, f"{DATA}/{t}.jsonl")
        print(f"  {t:14s} {zeroshot[t]}%", flush=True)

    mcq = ["test_medqa", "test_medmcqa", "test_mmlu"]
    comb_n = sum(SET_COUNTS[s] for s in mcq)
    comb = sum(zeroshot[s] * SET_COUNTS[s] for s in mcq) / comb_n
    print(f"\n组合 MCQ 零样本: {comb:.2f}%")
    print(f"DeepSeek 教师组合: 79.80%  -> headroom = {79.80 - comb:+.2f}pp")

    json.dump({"zeroshot": zeroshot, "combined": round(comb, 2)},
              open(f"{RUN}/eval_results_qwen3_zeroshot.json", "w"), ensure_ascii=False, indent=2)
    print(f"-> {RUN}/eval_results_qwen3_zeroshot.json")


if __name__ == "__main__":
    main()
