#!/usr/bin/env python3
"""Qwen3-32B thinking 模式零样本（600 题筛选集）。

- enable_thinking=True：模型先吐 <think>...</think> 推理链，答案在 </think> 之后。
- 慢：每题推理 ~800 token，600 题约 3-4h。仅 600 筛选集（与教师锚同口径）。
- 产出 runs/eval_results_qwen3_thinking_600.json。
"""
import json
import os
import sys

import torch

STUDENT = "models/Qwen3-32B"
DATA = "fullEnglish/00_data/out/screen_input.jsonl"
OUT = "fullEnglish/03_main_distill/runs/eval_results_qwen3_thinking_600.json"
os.environ["DISTILL_PROMPT_LANG"] = "en"
sys.path.insert(0, "shared")
from train_choice_head_distill import build_mcq_prompt, extract_answer_char  # noqa: E402


def extract_thinking_answer(gen):
    # 答案在 </think> 之后；兜底取最后一个 A-E 字母
    if "</think>" in gen:
        return extract_answer_char(gen.split("</think>", 1)[1])
    for ch in reversed(gen.strip().upper()):
        if ch in "ABCDE":
            return ch
    return ""


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(STUDENT, trust_remote_code=True)
    print(f"加载 {STUDENT} (bf16, enable_thinking=True) ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        STUDENT, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="cuda:0",
    )
    model.eval()

    rows = [json.loads(l) for l in open(DATA) if l.strip()]
    correct = total = 0
    src = {}
    for i, item in enumerate(rows):
        q = item["Question"]
        opts = item["Options"]
        gt = str(item["Answer"]).strip().upper()
        s = item.get("source", "?")
        sys_line, user_block = build_mcq_prompt(q, opts)
        msgs = [{"role": "system", "content": sys_line},
                {"role": "user", "content": user_block}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=True)
        inputs = tok(prompt, return_tensors="pt", truncation=True).to("cuda:0")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=2048, do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        gen = tok.decode(out[0][inputs["input_ids"].size(1):], skip_special_tokens=True)
        pred = extract_thinking_answer(gen)
        if pred == gt:
            correct += 1
        total += 1
        src.setdefault(s, [0, 0])
        src[s][0] += 1
        src[s][1] += int(pred == gt)
        if (i + 1) % 50 == 0:
            print(f"[{i+1}/{len(rows)}] acc={100.0*correct/total:.2f}%", flush=True)

    acc = round(100.0 * correct / total, 2) if total else 0.0
    print(f"\nQwen3-32B thinking 模式 600 题零样本: {acc}% ({correct}/{total})")
    for s, (n, c) in sorted(src.items()):
        print(f"  {s}: {100.0*c/n:.1f}% ({c}/{n})")
    json.dump({"acc": acc, "correct": correct, "total": total, "per_source": src},
              open(OUT, "w"), ensure_ascii=False, indent=2)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
