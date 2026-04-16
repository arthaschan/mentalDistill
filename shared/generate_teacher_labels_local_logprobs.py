#!/usr/bin/env python3
"""
从本地模型提取 ABCDE 真实概率分布，生成教师软标签。
输出同时包含 TeacherAnswer（硬标签）和 TeacherDist（真实概率分布），
可直接用于 choice-head 蒸馏，无需再经过 prepare_soft_labels.py。

支持:
  - 标准 HuggingFace 模型（BF16 / FP16）
  - AWQ / GPTQ 量化模型（自动检测）
  - vLLM 加速推理（可选）

用法:
    python generate_teacher_labels_local_logprobs.py \
        --model_path models/Llama-3.3-70B-Instruct-AWQ \
        --dataset 15_fulldata_resplit/data/train.jsonl \
        --output data/teacher_train_llama70b.jsonl \
        --resume
"""
import argparse
import json
import hashlib
import re
import sys
from pathlib import Path

import torch

OPTION_LETTERS = ["A", "B", "C", "D", "E"]
ANSWER_RE = re.compile(r"\b([A-E])\b")

SYSTEM_PROMPT = (
    "你是一位专业的医学专家。请根据你的专业知识回答以下单项选择题。"
    "只能输出一个大写字母（A/B/C/D/E），不要输出解释，不要输出多余文本。"
)


def build_question_text(item):
    q = str(item.get("Question") or item.get("question") or "").strip()
    options = item.get("Options") or item.get("options") or {}
    lines = [q]
    if isinstance(options, dict):
        for k in OPTION_LETTERS:
            if k in options:
                lines.append(f"{k}. {str(options[k]).strip()}")
    else:
        opt_text = str(options).strip()
        if opt_text:
            lines.append(opt_text)
    lines.append("请只输出一个大写字母（A/B/C/D/E）。")
    return "\n".join(lines)


def sample_key(item):
    q = str(item.get("Question") or item.get("question") or "").strip()
    options = item.get("Options") or item.get("options") or {}
    if isinstance(options, dict):
        opt_text = "\n".join(f"{k}:{str(options.get(k, ''))}" for k in OPTION_LETTERS)
    else:
        opt_text = str(options).strip()
    raw = f"{q}\n{opt_text}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def extract_answer(text):
    if not text:
        return ""
    t = text.strip().upper()
    if len(t) == 1 and t in "ABCDE":
        return t
    m = ANSWER_RE.search(t)
    return m.group(1) if m else ""


def get_option_token_ids(tokenizer):
    """获取 A/B/C/D/E 在词表中的 token ID。"""
    ids = {}
    for letter in OPTION_LETTERS:
        # 尝试多种编码方式，取最短的单 token
        candidates = []
        for text in [letter, f" {letter}", f"\n{letter}"]:
            encoded = tokenizer.encode(text, add_special_tokens=False)
            candidates.append(encoded)
        # 直接编码单字母
        direct = tokenizer.encode(letter, add_special_tokens=False)
        if len(direct) == 1:
            ids[letter] = direct[0]
        else:
            # fallback: 取 " A" 编码的最后一个 token
            space_encoded = tokenizer.encode(f" {letter}", add_special_tokens=False)
            ids[letter] = space_encoded[-1]
    return ids


def extract_logprobs_from_logits(logits, option_token_ids, temperature=1.0):
    """从 logits 中提取 ABCDE 的概率分布。

    Args:
        logits: shape (vocab_size,) 最后一个 token 位置的 logits
        option_token_ids: dict {letter: token_id}
        temperature: softmax 温度
    Returns:
        dict {letter: probability}
    """
    # 提取 5 个选项的 logits
    option_logits = torch.tensor(
        [logits[option_token_ids[k]].item() for k in OPTION_LETTERS],
        dtype=torch.float64
    )
    # 在 5 个选项上做 softmax（不是全词表 softmax）
    option_logits = option_logits / temperature
    probs = torch.softmax(option_logits, dim=0)
    return {k: round(probs[i].item(), 6) for i, k in enumerate(OPTION_LETTERS)}


def main():
    parser = argparse.ArgumentParser(
        description="从本地模型提取 ABCDE 真实概率分布作为教师软标签"
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="本地模型路径")
    parser.add_argument("--dataset", type=str, required=True,
                        help="输入 JSONL 数据集")
    parser.add_argument("--output", type=str, required=True,
                        help="输出 JSONL（含 TeacherDist）")
    parser.add_argument("--system_prompt", type=str, default=None,
                        help="系统提示词文件路径（默认使用内置通用 prompt）")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="logits softmax 温度（默认 1.0）")
    parser.add_argument("--resume", action="store_true",
                        help="断点续传：跳过已处理的样本")
    parser.add_argument("--gt_field", type=str, default="Answer",
                        help="GT 答案字段名")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 加载系统提示词
    system_prompt = SYSTEM_PROMPT
    if args.system_prompt:
        system_prompt = Path(args.system_prompt).read_text(encoding="utf-8").strip()

    # 加载数据
    rows = []
    with open(args.dataset, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    print(f"Loaded {len(rows)} samples from {args.dataset}")

    # 断点续传
    done_keys = set()
    if args.resume and out_path.exists():
        with open(out_path, "r", encoding="utf-8") as rf:
            for line in rf:
                line = line.strip()
                if line:
                    try:
                        row = json.loads(line)
                        done_keys.add(sample_key(row))
                    except json.JSONDecodeError:
                        continue
        print(f"Resume: {len(done_keys)} samples already processed, skipping.")

    # 加载模型
    print(f"Loading model from {model_path} ...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), trust_remote_code=True
    )

    # 检测是否为量化模型
    config_path = model_path / "config.json"
    is_quantized = False
    if config_path.exists():
        with open(config_path) as cf:
            config = json.load(cf)
        quant_config = config.get("quantization_config", {})
        quant_method = quant_config.get("quant_method", "")
        if quant_method in ("awq", "gptq"):
            is_quantized = True
            print(f"Detected {quant_method.upper()} quantized model")

    load_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
    }
    if not is_quantized:
        load_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(str(model_path), **load_kwargs)
    model.eval()
    print("Model loaded.")

    # 获取 ABCDE token IDs
    option_ids = get_option_token_ids(tokenizer)
    print(f"Option token IDs: {option_ids}")
    # 验证 token 映射正确性
    for letter, tid in option_ids.items():
        decoded = tokenizer.decode([tid]).strip()
        print(f"  {letter} -> token_id={tid} -> decoded='{decoded}'")

    # 推理
    total = 0
    parsed = 0
    correct = 0
    skipped = 0
    entropy_sum = 0.0

    mode = "a" if args.resume else "w"
    with open(args.output, mode, encoding="utf-8") as wf:
        for i, item in enumerate(rows):
            key = sample_key(item)
            if key in done_keys:
                skipped += 1
                continue

            user_text = build_question_text(item)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ]
            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )
                # logits shape: (1, seq_len, vocab_size)
                # 取最后一个 token 位置的 logits（即模型对下一个 token 的预测）
                last_logits = outputs.logits[0, -1, :]

            # 提取 ABCDE 真实概率分布
            teacher_dist = extract_logprobs_from_logits(
                last_logits, option_ids, temperature=args.temperature
            )

            # 硬标签 = 概率最高的选项
            teacher_ans = max(teacher_dist, key=teacher_dist.get)
            gt_ans = str(
                item.get(args.gt_field) or item.get("answer") or ""
            ).strip().upper()

            total += 1
            if teacher_ans in OPTION_LETTERS:
                parsed += 1
                if teacher_ans == gt_ans:
                    correct += 1

            # 计算熵（衡量教师的确定程度）
            import math
            entropy = -sum(
                p * math.log(p + 1e-12) for p in teacher_dist.values()
            )
            entropy_sum += entropy

            out_row = dict(item)
            out_row["TeacherAnswer"] = teacher_ans
            out_row["TeacherDist"] = teacher_dist
            out_row["TeacherEntropy"] = round(entropy, 4)
            out_row["OriginalAnswer"] = gt_ans
            out_row["Answer"] = teacher_ans
            wf.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            wf.flush()

            if (i + 1) % 50 == 0 or (i + 1) == len(rows):
                acc = correct / total * 100 if total else 0
                avg_ent = entropy_sum / total if total else 0
                print(
                    f"  [{i+1}/{len(rows)}] "
                    f"processed={total} correct={correct}/{total} ({acc:.1f}%) "
                    f"avg_entropy={avg_ent:.3f}"
                )

    acc = correct / total * 100 if total else 0
    mismatch = parsed - correct
    avg_entropy = entropy_sum / total if total else 0

    print(f"\n{'='*50}")
    print(f"Teacher Label Summary (with real logprobs)")
    print(f"{'='*50}")
    print(f"Model:          {model_path.name}")
    print(f"Total:          {total}")
    print(f"Parsed:         {parsed}")
    print(f"Correct:        {correct}")
    print(f"Teacher Acc:    {acc:.2f}%")
    print(f"Mismatch:       {mismatch} ({mismatch/total*100:.2f}%)")
    print(f"Avg Entropy:    {avg_entropy:.4f}")
    print(f"Skipped(resume):{skipped}")
    print(f"Output:         {args.output}")
    print(f"\nNote: TeacherDist contains REAL probability distribution,")
    print(f"      no label smoothing needed. Feed directly to distillation.")


if __name__ == "__main__":
    main()
