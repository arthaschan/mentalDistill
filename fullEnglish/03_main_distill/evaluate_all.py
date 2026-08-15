#!/usr/bin/env python3
"""fullEnglish — 评估所有训练 adapter (每个 adapter 加载一次模型, 跑 4 个测试集).

在 03_main_distill/runs/ 下扫描每个 32B_*_s* 训练目录, 取 best/ (val 最优) adapter,
对 test_medqa / test_medmcqa / test_mmlu / test_pubmedqa 做确定性评估,
连同学生零样本地板一起写入 runs/eval_results.json.
"""
import argparse
import glob
import json
import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LETTERS = ["A", "B", "C", "D", "E"]
SYSTEM_LINE = ("You are a medical expert. Output exactly one letter "
               "(A, B, C, D, or E) as the answer, with no explanation or spaces.\n")

TEST_SETS = ["test_medqa", "test_medmcqa", "test_mmlu", "test_pubmedqa"]


def build_prompt(q, opts):
    return ("<|im_start|>system\n" + SYSTEM_LINE + "<|im_end|>\n"
            "<|im_start|>user\n" + f"Question: {q}\nOptions:\n{opts}\n" + "<|im_end|>\n"
            "<|im_start|>assistant\n")


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


def eval_set(model, tokenizer, device, path):
    rows = load_jsonl(path)
    correct = total = 0
    for item in rows:
        q = item.get("Question", "")
        opts = item.get("Options", "")
        gt = str(item.get("Answer", "")).strip().upper()
        if not q or not opts or gt not in LETTERS:
            continue
        total += 1
        prompt = build_prompt(q, opts)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=4, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        gen = tokenizer.decode(out[0][inputs["input_ids"].size(1):], skip_special_tokens=True)
        if extract_answer_char(gen) == gt:
            correct += 1
    return round(100.0 * correct / total, 2) if total else 0.0


def load_model(base_model, adapter_dir, device):
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if adapter_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return model, tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default=None, help="默认 fullEnglish/03_main_distill/runs")
    ap.add_argument("--data_dir", default=None, help="默认 fullEnglish/00_data/out")
    ap.add_argument("--student", default=None, help="默认 $BASE_MODEL_32B 或 models/Qwen2.5-32B-Instruct")
    args = ap.parse_args()

    FE = os.path.dirname(os.path.abspath(__file__))
    RUN = args.run_dir or os.path.join(FE, "runs")
    DATA = args.data_dir or os.path.join(os.path.dirname(FE), "00_data", "out")
    STUDENT = args.student or os.environ.get("BASE_MODEL_32B") or "models/Qwen2.5-32B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = {"student": STUDENT, "zeroshot": {}, "adapters": {}}

    # 学生零样本地板 (无 adapter)
    print(f"=== 学生零样本地板: {STUDENT} ===")
    model, tokenizer = load_model(STUDENT, None, device)
    for t in TEST_SETS:
        acc = eval_set(model, tokenizer, device, os.path.join(DATA, f"{t}.jsonl"))
        results["zeroshot"][t] = acc
        print(f"  zeroshot_{t:14s} {acc:.2f}%")
    del model
    torch.cuda.empty_cache()

    # 每个 adapter (best/)
    adapters = sorted(glob.glob(os.path.join(RUN, "32B_*_s*", "best")))
    for ad in adapters:
        name = os.path.basename(os.path.dirname(ad))
        print(f"=== adapter: {name} ===")
        model, tokenizer = load_model(STUDENT, ad, device)
        accs = {}
        for t in TEST_SETS:
            acc = eval_set(model, tokenizer, device, os.path.join(DATA, f"{t}.jsonl"))
            accs[t] = acc
            print(f"  {t:16s} {acc:.2f}%")
        results["adapters"][name] = {"path": ad, "acc": accs}
        del model
        torch.cuda.empty_cache()

    out_path = os.path.join(RUN, "eval_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
