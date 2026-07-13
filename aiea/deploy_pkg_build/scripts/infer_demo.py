#!/usr/bin/env python3
"""单题推理示例：加载 Qwen2.5-14B 基座 + Stage-1 Choice-Head LoRA adapter，
对一道选择题确定性预测 A/B/C/D/E。与评估脚本同口径（同 system prompt、贪婪解码）。

用法:
    python scripts/infer_demo.py --base_model /data/models/Qwen2.5-14B-Instruct
"""
import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

OPTION_LETTERS = ["A", "B", "C", "D", "E"]

# 示例题（可替换为你自己的题目）
DEMO_ITEM = {
    "Question": "全口义齿排列时，上颌第二双尖牙舌尖与（牙合）平面的关系",
    "Options": "A 与（牙合）平面接触\nB 离开（牙合）平面0.5mm\nC 离开（牙合）平面1.0mm\nD 离开平面1.5mm\nE 离开（牙合）平面2mm",
}


def extract_answer_char(text: str) -> str:
    for ch in text.strip().upper():
        if ch in OPTION_LETTERS:
            return ch
    return ""


def build_prompt(item: dict) -> str:
    messages = [
        {"role": "system", "content": "你是一位专业的牙科医生。请根据你的专业知识回答以下选择题，只输出一个大写字母（A/B/C/D/E）。"},
        {"role": "user", "content": f"{item.get('Question','')}\n{item.get('Options','')}\n请只输出一个大写字母作为答案。"},
    ]
    prompt = ""
    for m in messages:
        prompt += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    return prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True, help="Qwen2.5-14B-Instruct 基座绝对路径")
    ap.add_argument("--adapter_dir", default=str(Path(__file__).resolve().parent.parent / "adapter"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    print(f"加载基础模型: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16,
        device_map=args.device, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    print(f"加载 LoRA adapter: {args.adapter_dir}")
    model = PeftModel.from_pretrained(model, args.adapter_dir)
    model.eval()

    prompt = build_prompt(DEMO_ITEM)
    inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=4, do_sample=False, temperature=1.0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    print("\n题目:", DEMO_ITEM["Question"])
    print("选项:\n" + DEMO_ITEM["Options"])
    print("模型原始输出:", repr(resp.strip()))
    print("预测答案:", extract_answer_char(resp))


if __name__ == "__main__":
    main()
