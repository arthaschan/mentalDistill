#!/usr/bin/env python3
"""推理链蒸馏 SFT 训练：让学生模型学习教师的推理过程。

输入：train_cot.jsonl（包含 Rationale 字段的 CoT 数据）
训练目标：Question + Options → Rationale（推理过程 + 答案）

用法:
    python train_rationale_sft.py \
        --base_model /path/to/Qwen2.5-7B-Instruct \
        --train_data data/train_cot.jsonl \
        --val_data data/val.jsonl \
        --output_dir outputs/rationale_sft \
        --epochs 6 --lr 1e-4 --seed 42
"""
import argparse
import json
import os
import random
import re

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model

OPTION_LETTERS = ["A", "B", "C", "D", "E"]

RATIONALE_SYSTEM_PROMPT = (
    "你是一位专业的牙科医生。请对以下选择题进行分析推理，逐步分析各选项，"
    "然后在最后一行以\"答案：X\"结尾给出答案（X为A/B/C/D/E）。"
)

# Evaluation uses the short answer-only prompt for consistency with other modules
EVAL_SYSTEM_PROMPT = (
    "你是一位专业的牙科医生。请根据你的专业知识回答以下选择题，只输出一个大写字母（A/B/C/D/E）。"
)


def set_global_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class RationaleSFTDataset(Dataset):
    """训练数据集：问题 → 推理链 + 答案"""

    def __init__(self, data_path, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = load_jsonl(data_path)
        print(f"加载推理链 SFT 数据集: {len(self.data)} 条")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item.get("Question", "")
        options = item.get("Options", "")
        rationale = item.get("Rationale", "")

        # Build ChatML prompt
        prompt_prefix = (
            f"<|im_start|>system\n{RATIONALE_SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n问题：{question}\n选项：\n{options}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        full_text = prompt_prefix + rationale + "<|im_end|>"

        inputs = self.tokenizer(
            full_text, truncation=True, padding="max_length",
            max_length=self.max_length, return_tensors="pt",
        )

        # Label masking: only compute loss on the rationale (assistant response)
        prefix_enc = self.tokenizer(prompt_prefix, truncation=True, max_length=self.max_length)
        prefix_len = len(prefix_enc["input_ids"])

        labels = inputs["input_ids"].squeeze().clone()
        labels[:prefix_len] = -100
        labels[inputs["attention_mask"].squeeze() == 0] = -100

        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": labels,
        }


def extract_answer_char(text):
    """从生成文本中提取答案字母。"""
    # 先尝试 '答案：X' 格式
    m = re.search(r"答案[：:]\s*([A-E])", text.upper())
    if m:
        return m.group(1)
    # Fallback: 文本中第一个 A-E 字母
    for ch in text.strip().upper():
        if ch in OPTION_LETTERS:
            return ch
    return ""


def build_eval_prompt(item):
    """构建评估用的简短 prompt（与其他模块一致）。"""
    q = item.get("Question", "")
    opts = item.get("Options", "")
    prompt = (
        f"<|im_start|>system\n{EVAL_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{q}\n{opts}\n请只输出一个大写字母作为答案。<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return prompt


def evaluate_val(model, tokenizer, val_data, device="cuda"):
    """在验证集上评估准确率（短答案模式，与其他模块一致）。"""
    model.eval()
    correct = 0
    total = 0
    for item in val_data:
        gt = item.get("Answer", "").strip().upper()
        if not gt:
            continue
        total += 1
        prompt = build_eval_prompt(item)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=4, do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                     skip_special_tokens=True)
        pred = extract_answer_char(response)
        if pred == gt[0].upper():
            correct += 1
    acc = correct / total * 100 if total > 0 else 0
    model.train()
    return acc


def main():
    parser = argparse.ArgumentParser(description="推理链蒸馏 SFT 训练")
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--val_data", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--gradient_accumulation", type=int, default=1)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    set_global_seed(args.seed, args.deterministic)

    output_dir = Path(args.output_dir)
    best_dir = output_dir / "best"
    output_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)

    # Load model and tokenizer
    print(f"加载基础模型: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # LoRA
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Data
    train_dataset = RationaleSFTDataset(args.train_data, tokenizer, args.max_length)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              drop_last=True)
    val_data = load_jsonl(args.val_data)

    # Optimizer & scheduler
    total_steps = len(train_loader) * args.epochs // args.gradient_accumulation
    warmup_steps = int(total_steps * args.warmup_ratio)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Training loop
    best_val = 0.0
    device = "cuda"
    model.train()

    for epoch in range(1, args.epochs + 1):
        pbar = tqdm(train_loader, desc=f"Rationale SFT Epoch {epoch}/{args.epochs}")
        epoch_loss = 0.0
        step_count = 0

        for batch_idx, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                            labels=labels)
            loss = outputs.loss / args.gradient_accumulation
            loss.backward()

            if (batch_idx + 1) % args.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * args.gradient_accumulation
            step_count += 1
            pbar.set_postfix({"loss": f"{loss.item() * args.gradient_accumulation:.4f}"})

        avg_loss = epoch_loss / max(step_count, 1)
        val_acc = evaluate_val(model, tokenizer, val_data, device)
        print(f"[VAL] epoch={epoch} avg_loss={avg_loss:.4f} acc={val_acc:.2f}%")

        if val_acc >= best_val:
            best_val = val_acc
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            print(f"  -> 保存最佳模型 val_acc={val_acc:.2f}%")

    # Save final
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    print(f"\n训练完成. best_val={best_val:.2f}%")
    print(f"最佳模型: {best_dir}")
    print(f"最终模型: {final_dir}")


if __name__ == "__main__":
    main()
