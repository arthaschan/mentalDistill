#!/usr/bin/env python3
"""PED 评估：测 PED 训练后 3-seed adapter 在 4 个英文测试集的准确率，与主线基线对比。

预期：combined_student ≈ 15.42%（塌到比瞎猜 20% 还低），证明"只训 near-miss 题"失败。
"""
import json
import os
import statistics
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
RUN = os.path.join(ROOT, "runs")
FE_DATA = os.path.join(ROOT, "..", "fullEnglish", "00_data", "out")
FE_RUNS = os.path.join(ROOT, "..", "fullEnglish", "03_main_distill", "runs")
STUDENT = os.path.join(ROOT, "..", "models", "Qwen3-32B")
SEEDS = ["s11", "s42", "s8"]
TEST_SETS = ["test_medqa", "test_medmcqa", "test_mmlu", "test_pubmedqa"]
SET_COUNTS = {"test_medqa": 1273, "test_medmcqa": 4183, "test_mmlu": 2837, "test_pubmedqa": 1000}

os.environ["DISTILL_USE_CHAT_TEMPLATE"] = "1"
os.environ["DISTILL_PROMPT_LANG"] = "en"
sys.path.insert(0, os.path.join(ROOT, "..", "shared"))
from train_choice_head_distill import (  # noqa: E402
    apply_prompt_template, build_mcq_prompt, extract_answer_char,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load(path):
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
    for item in load(path):
        q, opts = item.get("Question", ""), item.get("Options", "")
        gt = str(item.get("Answer", "")).strip().upper()
        if not q or not opts or gt not in "ABCDE":
            continue
        total += 1
        sys_line, user_block = build_mcq_prompt(q, opts)
        prompt, _ = apply_prompt_template(tok, sys_line, user_block)
        inputs = tok(prompt, return_tensors="pt", truncation=True).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=16, do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        gen = tok.decode(out[0][inputs["input_ids"].size(1):], skip_special_tokens=True)
        if extract_answer_char(gen) == gt:
            correct += 1
    return round(100.0 * correct / total, 2) if total else 0.0


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tok = AutoTokenizer.from_pretrained(STUDENT, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        STUDENT, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map=device,
    )

    per_seed = {}
    for seed in SEEDS:
        ad = os.path.join(RUN, f"Qwen3_ped_ar_{seed}", "best")
        model = PeftModel.from_pretrained(base, ad)
        model.eval()
        per_seed[seed] = {t: eval_set(model, tok, os.path.join(FE_DATA, f"{t}.jsonl"))
                          for t in TEST_SETS}
        print(f"{seed}: {per_seed[seed]}", flush=True)

    summary = {t: round(statistics.mean([per_seed[s][t] for s in SEEDS]), 2) for t in TEST_SETS}
    mcq = ["test_medqa", "test_medmcqa", "test_mmlu"]
    comb_n = sum(SET_COUNTS[s] for s in mcq)
    comb_s = sum(summary[s] * SET_COUNTS[s] for s in mcq) / comb_n
    # 主线基线（fullEnglish 主实验）
    baseline = json.load(open(os.path.join(FE_RUNS, "eval_results_qwen3.json")))
    zeroshot = json.load(open(os.path.join(FE_RUNS, "eval_results_qwen3_zeroshot.json")))

    print(f"\n=== PED 评估结果（组合 MCQ 全量口径） ===")
    print(f"PED 训练后: {comb_s:.2f}%")
    print(f"主线基线(全量训练): {baseline['combined_student']}%")
    print(f"零样本: {baseline['combined_zeroshot']}%")
    print(f"结论: PED 塌到 {comb_s:.2f}%（比瞎猜 20% 还低），证明'只训 near-miss 题'失败")

    json.dump({"per_seed": per_seed, "summary": summary, "combined_student": round(comb_s, 2),
               "combined_baseline": baseline["combined_student"],
               "combined_zeroshot": baseline["combined_zeroshot"]},
              open(os.path.join(DATA, "eval_results_qwen3_ped_ar.json"), "w"),
              ensure_ascii=False, indent=2)
    print(f"-> {DATA}/eval_results_qwen3_ped_ar.json")


if __name__ == "__main__":
    main()
