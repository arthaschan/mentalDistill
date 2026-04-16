#!/usr/bin/env python3
"""
使用 vLLM 从本地模型提取 ABCDE 真实概率分布，生成教师软标签。
输出同时包含 TeacherAnswer（硬标签）和 TeacherDist（真实概率分布）。

vLLM 原生支持 AWQ/GPTQ 量化模型，速度远快于 HuggingFace 逐条推理。

用法:
    python generate_teacher_labels_vllm.py \
        --model_path models/Llama-3.3-70B-Instruct-AWQ \
        --dataset 15_fulldata_resplit/data/train.jsonl \
        --output data/teacher_train_llama70b.jsonl \
        --resume
"""
import argparse
import json
import hashlib
import math
import re
import sys
from pathlib import Path

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


def main():
    parser = argparse.ArgumentParser(
        description="使用 vLLM 从本地模型提取 ABCDE 真实概率分布"
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--system_prompt", type=str, default=None)
    parser.add_argument("--gt_field", type=str, default="Answer")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="vLLM batch size for prompt processing")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    args = parser.parse_args()

    model_path = Path(args.model_path)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    system_prompt = SYSTEM_PROMPT
    if args.system_prompt:
        system_prompt = Path(args.system_prompt).read_text(encoding="utf-8").strip()

    # Load dataset
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

    # Resume
    done_keys = set()
    existing_results = []
    if args.resume and out_path.exists():
        with open(out_path, "r", encoding="utf-8") as rf:
            for line in rf:
                line = line.strip()
                if line:
                    try:
                        row = json.loads(line)
                        done_keys.add(sample_key(row))
                        existing_results.append(row)
                    except json.JSONDecodeError:
                        continue
        print(f"Resume: {len(done_keys)} samples already done, skipping.")

    todo = [(i, item) for i, item in enumerate(rows) if sample_key(item) not in done_keys]
    if not todo:
        print("All samples already processed. Nothing to do.")
        # Still print summary from existing results
        _print_summary(existing_results, args.gt_field, model_path.name, len(done_keys))
        return

    print(f"Need to process: {len(todo)} samples")

    # Load vLLM
    print(f"Loading model via vLLM from {model_path} ...")
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=str(model_path),
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=4096,
        quantization="awq",
    )
    tokenizer = llm.get_tokenizer()

    # Get option token IDs
    option_ids = {}
    for letter in OPTION_LETTERS:
        tids = tokenizer.encode(letter, add_special_tokens=False)
        if len(tids) == 1:
            option_ids[letter] = tids[0]
        else:
            space_tids = tokenizer.encode(f" {letter}", add_special_tokens=False)
            option_ids[letter] = space_tids[-1]

    print(f"Option token IDs: {option_ids}")
    for letter, tid in option_ids.items():
        decoded = tokenizer.decode([tid]).strip()
        print(f"  {letter} -> token_id={tid} -> decoded='{decoded}'")

    # Build prompts using chat template
    prompts = []
    for _, item in todo:
        user_text = build_question_text(item)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt)

    # Run vLLM inference with logprobs
    sampling_params = SamplingParams(
        max_tokens=1,
        temperature=0,
        logprobs=20,  # get top-20 logprobs to ensure ABCDE are covered
    )

    print(f"Running vLLM inference on {len(prompts)} prompts ...")
    outputs = llm.generate(prompts, sampling_params)
    print(f"Inference done. Processing results ...")

    # Process results
    results = []
    for idx, ((orig_i, item), output) in enumerate(zip(todo, outputs)):
        # Extract logprobs from first generated token
        token_logprobs = output.outputs[0].logprobs[0] if output.outputs[0].logprobs else {}

        # Build ABCDE distribution from logprobs
        import torch
        option_logprob_vals = []
        for letter in OPTION_LETTERS:
            tid = option_ids[letter]
            if tid in token_logprobs:
                option_logprob_vals.append(token_logprobs[tid].logprob)
            else:
                option_logprob_vals.append(-100.0)  # very low default

        # Softmax over 5 options
        logits_t = torch.tensor(option_logprob_vals, dtype=torch.float64)
        probs = torch.softmax(logits_t, dim=0)
        teacher_dist = {k: round(probs[i].item(), 6) for i, k in enumerate(OPTION_LETTERS)}

        teacher_ans = max(teacher_dist, key=teacher_dist.get)
        gt_ans = str(item.get(args.gt_field) or item.get("answer") or "").strip().upper()

        entropy = -sum(p * math.log(p + 1e-12) for p in teacher_dist.values())

        out_row = dict(item)
        out_row["TeacherAnswer"] = teacher_ans
        out_row["TeacherDist"] = teacher_dist
        out_row["TeacherEntropy"] = round(entropy, 4)
        out_row["OriginalAnswer"] = gt_ans
        out_row["Answer"] = teacher_ans
        results.append(out_row)

    # Write output (overwrite or append)
    mode = "w"
    all_results = existing_results + results
    with open(out_path, mode, encoding="utf-8") as wf:
        for row in all_results:
            wf.write(json.dumps(row, ensure_ascii=False) + "\n")

    _print_summary(all_results, args.gt_field, model_path.name, len(done_keys))


def _print_summary(all_results, gt_field, model_name, skipped):
    total = len(all_results)
    correct = 0
    parsed = 0
    entropy_sum = 0.0
    high_entropy = 0  # entropy > 1.0
    for row in all_results:
        t_ans = row.get("TeacherAnswer", "")
        gt_ans = row.get("OriginalAnswer", "") or str(row.get(gt_field, "")).strip().upper()
        ent = row.get("TeacherEntropy", 0)
        if t_ans in OPTION_LETTERS:
            parsed += 1
            if t_ans == gt_ans:
                correct += 1
        entropy_sum += ent
        if ent > 1.0:
            high_entropy += 1

    acc = correct / total * 100 if total else 0
    mismatch = parsed - correct
    avg_entropy = entropy_sum / total if total else 0

    print(f"\n{'='*55}")
    print(f"  Teacher Label Summary (vLLM + real logprobs)")
    print(f"{'='*55}")
    print(f"  Model:             {model_name}")
    print(f"  Total:             {total}")
    print(f"  Correct:           {correct}")
    print(f"  Teacher Accuracy:  {acc:.2f}%")
    print(f"  Mismatch (T≠GT):  {mismatch} ({mismatch/total*100:.2f}%)")
    print(f"  Avg Entropy:       {avg_entropy:.4f}")
    print(f"  High Entropy (>1): {high_entropy} ({high_entropy/total*100:.1f}%)")
    print(f"  Skipped (resume):  {skipped}")
    print(f"{'='*55}")
    print(f"  TeacherDist = REAL probability distribution.")
    print(f"  No label smoothing needed.")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
