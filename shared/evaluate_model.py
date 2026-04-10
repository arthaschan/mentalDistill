#!/usr/bin/env python3
"""通用模型评估脚本：加载 base + LoRA adapter，在测试集上做确定性推理并计算准确率。

用法:
    python evaluate_model.py --base_model ./Qwen2.5-7B-Instruct \
        --adapter_dir ./output/best \
        --test_data ./data/test.jsonl \
        [--wrong_log ./output/test_wrong.jsonl]
"""
import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

OPTION_LETTERS = ["A", "B", "C", "D", "E"]


def extract_answer_char(text: str) -> str:
    for ch in text.strip().upper():
        if ch in OPTION_LETTERS:
            return ch
    return ""


def build_prompt(item: dict) -> str:
    q = item.get("Question", "")
    opts = item.get("Options", "")
    messages = [
        {"role": "system", "content": "你是一位专业的牙科医生。请根据你的专业知识回答以下选择题，只输出一个大写字母（A/B/C/D/E）。"},
        {"role": "user", "content": f"{q}\n{opts}\n请只输出一个大写字母作为答案。"},
    ]
    # ChatML format
    prompt = ""
    for m in messages:
        prompt += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    return prompt


def load_jsonl(path: str) -> list:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def evaluate(base_model: str, adapter_dir: str, test_data: str,
             wrong_log: str = None, device: str = "cuda"):
    print(f"加载基础模型: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map=device,
        trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    if adapter_dir:
        print(f"加载 LoRA adapter: {adapter_dir}")
        model = PeftModel.from_pretrained(model, adapter_dir)

    model.eval()
    samples = load_jsonl(test_data)

    correct = 0
    total = 0
    wrong_items = []

    for item in samples:
        gt = item.get("Answer", "").strip().upper()
        if not gt:
            continue
        total += 1
        prompt = build_prompt(item)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=4,
                do_sample=False, temperature=1.0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)

        response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                     skip_special_tokens=True)
        pred = extract_answer_char(response)

        if pred == gt[0].upper():
            correct += 1
        else:
            wrong_items.append({
                "Question": item.get("Question", ""),
                "Options": item.get("Options", ""),
                "Answer": gt,
                "Predicted": pred,
                "Raw": response.strip(),
            })

    acc = correct / total * 100 if total > 0 else 0
    print(f"\n测试集准确率: {acc:.2f}% ({correct}/{total})")

    if wrong_log and wrong_items:
        Path(wrong_log).parent.mkdir(parents=True, exist_ok=True)
        with open(wrong_log, "w", encoding="utf-8") as f:
            for item in wrong_items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"错误样本已记录在 {wrong_log}")

    return acc, correct, total


def main():
    parser = argparse.ArgumentParser(description="模型评估")
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter_dir", default=None,
                        help="LoRA adapter 路径，不指定则评估原始模型")
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--wrong_log", default=None)
    args = parser.parse_args()
    evaluate(args.base_model, args.adapter_dir, args.test_data, args.wrong_log)


if __name__ == "__main__":
    main()
