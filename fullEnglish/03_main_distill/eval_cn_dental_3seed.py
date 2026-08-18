#!/usr/bin/env python3
"""中文牙科 Qwen3-32B 学生 3-seed 评估（test_dental 125 题），与 DeepSeek 老师 79.20% 对比。"""
import json
import os
import statistics
import sys

import torch

CN_DATA = "15_fulldata_resplit/data"
TEST = f"{CN_DATA}/test_dental.jsonl"
RUN = "fullEnglish/03_main_distill/runs"
STUDENT = "models/Qwen3-32B"
SEEDS = ["s11", "s42", "s8"]
TEACHER_ACC = 79.20

os.environ["DISTILL_USE_CHAT_TEMPLATE"] = "1"
os.environ["DISTILL_PROMPT_LANG"] = "zh"
sys.path.insert(0, "shared")
from train_choice_head_distill import (  # noqa: E402
    apply_prompt_template, build_mcq_prompt, extract_answer_char,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = [json.loads(l) for l in open(TEST) if l.strip()]
    tok = AutoTokenizer.from_pretrained(STUDENT, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        STUDENT, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map=device,
    )

    def eval_adapter(ad):
        from peft import PeftModel
        model = PeftModel.from_pretrained(base, ad)
        model.eval()
        c = 0
        for r in rows:
            q, opts = r.get("Question", ""), r.get("Options", "")
            gt = str(r.get("Answer", "")).strip().upper()
            if not q or not opts or gt not in "ABCDE":
                continue
            sys_line, user_block = build_mcq_prompt(q, opts)
            prompt, _ = apply_prompt_template(tok, sys_line, user_block)
            inputs = tok(prompt, return_tensors="pt", truncation=True).to(device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=16, do_sample=False,
                                     pad_token_id=tok.pad_token_id or tok.eos_token_id)
            gen = tok.decode(out[0][inputs["input_ids"].size(1):], skip_special_tokens=True)
            if extract_answer_char(gen) == gt:
                c += 1
        return round(100.0 * c / len(rows), 2)

    per = {}
    for s in SEEDS:
        ad = f"{RUN}/Qwen3_cn_a00_{s}/best"
        per[s] = eval_adapter(ad)
        print(f"  {s}: {per[s]}%", flush=True)

    m = statistics.mean(per.values())
    sd = statistics.stdev(per.values())
    print(f"\n=== Qwen3-32B 中文牙科 test_dental(125) 3-seed ===")
    print(f"零样本: 77.6%  训练后: {m:.2f}±{sd:.2f}%  ({per})")
    print(f"老师 DeepSeek: {TEACHER_ACC}%")
    print(f"Δ(学生-老师): {m - TEACHER_ACC:+.2f}pp  -> {'超越' if m > TEACHER_ACC else '未超越'}")
    print(f"增益(零样本→训练后): {m - 77.6:+.2f}pp")

    json.dump({"teacher": TEACHER_ACC, "zeroshot": 77.6, "per_seed": per,
               "mean": round(m, 2), "std": round(sd, 2),
               "delta": round(m - TEACHER_ACC, 2), "gain": round(m - 77.6, 2)},
              open(f"{RUN}/eval_results_qwen3_cn_dental.json", "w"), ensure_ascii=False, indent=2)
    print(f"-> {RUN}/eval_results_qwen3_cn_dental.json")


if __name__ == "__main__":
    main()
