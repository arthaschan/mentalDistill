#!/usr/bin/env python3
"""多 Seed 集成推理：加载多个 LoRA adapter，逐样本推理后多数投票。

用法:
    python ensemble_majority_vote.py \
        --base_model ./Qwen2.5-7B-Instruct \
        --adapter_dirs seed_7/best seed_8/best seed_42/best ... \
        --test_data ./data/test.jsonl \
        --output ./ensemble_results.json
"""
import argparse
import json
from collections import Counter
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


def predict_with_adapter(base_model_obj, tokenizer, adapter_dir, samples, device="cuda"):
    """加载一个 adapter，推理全部样本，返回预测列表。"""
    print(f"  加载 adapter: {adapter_dir}")
    model = PeftModel.from_pretrained(base_model_obj, adapter_dir)
    model.eval()

    preds = []
    for item in samples:
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
        preds.append(pred)

    # 卸载 adapter 释放显存
    del model
    torch.cuda.empty_cache()
    return preds


def majority_vote(all_preds: list[list[str]]) -> list[str]:
    """对每个样本取多数投票。平票时取第一模型的预测。"""
    n_samples = len(all_preds[0])
    results = []
    for i in range(n_samples):
        votes = [preds[i] for preds in all_preds]
        counter = Counter(votes)
        winner = counter.most_common(1)[0][0]
        results.append(winner)
    return results


def main():
    parser = argparse.ArgumentParser(description="多 Seed 集成推理 (Majority Vote)")
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter_dirs", nargs="+", required=True,
                        help="各 seed 训练的 best adapter 路径列表")
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--output", default=None, help="输出结果 JSON 路径")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    samples = load_jsonl(args.test_data)

    print(f"加载基础模型: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map=device,
        trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    # 逐 adapter 推理
    all_preds = []
    individual_accs = []
    gt_answers = [item.get("Answer", "").strip().upper()[0] for item in samples if item.get("Answer")]
    n_total = len(gt_answers)

    for adapter_dir in args.adapter_dirs:
        preds = predict_with_adapter(base_model, tokenizer, adapter_dir, samples, device)
        all_preds.append(preds)
        correct = sum(1 for p, g in zip(preds, gt_answers) if p == g)
        acc = correct / n_total * 100
        individual_accs.append(acc)
        seed_name = Path(adapter_dir).parent.name
        print(f"  [{seed_name}] {acc:.2f}% ({correct}/{n_total})")

    # 多数投票
    ensemble_preds = majority_vote(all_preds)
    ensemble_correct = sum(1 for p, g in zip(ensemble_preds, gt_answers) if p == g)
    ensemble_acc = ensemble_correct / n_total * 100

    print(f"\n{'='*50}")
    print(f"集成结果 (Majority Vote, {len(args.adapter_dirs)} models)")
    print(f"{'='*50}")
    for i, adapter_dir in enumerate(args.adapter_dirs):
        seed_name = Path(adapter_dir).parent.name
        print(f"  {seed_name}: {individual_accs[i]:.2f}%")
    print(f"  平均: {sum(individual_accs)/len(individual_accs):.2f}%")
    print(f"  集成: {ensemble_acc:.2f}% ({ensemble_correct}/{n_total})")
    print(f"{'='*50}")

    # 保存详细结果
    if args.output:
        result = {
            "ensemble_acc": ensemble_acc,
            "ensemble_correct": ensemble_correct,
            "total": n_total,
            "n_models": len(args.adapter_dirs),
            "individual": [],
            "per_sample": [],
        }
        for i, adapter_dir in enumerate(args.adapter_dirs):
            result["individual"].append({
                "adapter": adapter_dir,
                "acc": individual_accs[i],
            })
        for j in range(n_total):
            votes = [all_preds[i][j] for i in range(len(all_preds))]
            result["per_sample"].append({
                "question_idx": j,
                "gt": gt_answers[j],
                "votes": votes,
                "ensemble_pred": ensemble_preds[j],
                "correct": ensemble_preds[j] == gt_answers[j],
            })

        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"详细结果已保存: {args.output}")


if __name__ == "__main__":
    main()
