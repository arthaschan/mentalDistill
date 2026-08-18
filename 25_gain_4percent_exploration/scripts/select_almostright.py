#!/usr/bin/env python3
"""PED 第一步：算 Qwen3-32B 在英文训练集(20488题)的零样本 ABCDE 分布，筛"差点答对"(near-miss)题。

- near-miss = 学生 top1 答错、但正确答案是 top2。
- 产出：① 全量 logprobs（qwen3_train_logprobs.jsonl）② near-miss 选题集（train_head_almostright.jsonl）。
"""
import json
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
TRAIN = os.path.join(ROOT, "..", "fullEnglish", "00_data", "out", "train.jsonl")
OUT_LOGPROBS = os.path.join(DATA, "qwen3_train_logprobs.jsonl")
OUT_SELECT = os.path.join(DATA, "train_head_almostright.jsonl")
STUDENT = os.path.join(ROOT, "..", "models", "Qwen3-32B")
LETTERS = ["A", "B", "C", "D", "E"]

os.environ["DISTILL_PROMPT_LANG"] = "en"
sys.path.insert(0, os.path.join(ROOT, "..", "shared"))
from train_choice_head_distill import build_mcq_prompt  # noqa: E402


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(STUDENT, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        STUDENT, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="cuda:0",
    )
    model.eval()
    option_ids = {}
    for letter in LETTERS:
        direct = tok.encode(letter, add_special_tokens=False)
        option_ids[letter] = direct[0] if len(direct) == 1 else tok.encode(f" {letter}", add_special_tokens=False)[-1]

    rows = [json.loads(l) for l in open(TRAIN) if l.strip()]
    stats = {"total": 0, "top1_correct": 0, "almost_right": 0, "far_wrong": 0}
    os.makedirs(DATA, exist_ok=True)
    with open(OUT_LOGPROBS, "w") as wl, open(OUT_SELECT, "w") as ws:
        for i, item in enumerate(rows):
            q = item.get("Question", "")
            opts = item.get("Options", "")
            gt = str(item.get("Answer", "")).strip().upper()
            if not q or not opts or gt not in LETTERS:
                continue
            sys_line, user_block = build_mcq_prompt(q, opts)
            msgs = [{"role": "system", "content": sys_line},
                    {"role": "user", "content": user_block}]
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
            inputs = tok(prompt, return_tensors="pt", truncation=True).to("cuda:0")
            with torch.no_grad():
                logits = model(**inputs).logits[0, -1, :]
            opt_logits = torch.tensor([logits[option_ids[k]].item() for k in LETTERS],
                                      dtype=torch.float64)
            probs = torch.softmax(opt_logits, dim=0)
            dist = {k: round(probs[j].item(), 6) for j, k in enumerate(LETTERS)}
            ranked = sorted(dist, key=lambda k: -dist[k])
            top1, top2 = ranked[0], ranked[1]

            stats["total"] += 1
            if top1 == gt:
                stats["top1_correct"] += 1
            elif top2 == gt:
                stats["almost_right"] += 1
            else:
                stats["far_wrong"] += 1

            rec = dict(item)
            rec["TeacherDist"] = dist
            rec["TeacherAnswer"] = top1
            rec["OriginalAnswer"] = gt
            rec["Top2"] = top2
            wl.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if top1 != gt and top2 == gt:
                sel = dict(item)
                sel["Answer"] = gt
                ws.write(json.dumps(sel, ensure_ascii=False) + "\n")

            if (i + 1) % 2000 == 0:
                print(f"  [{i+1}/{len(rows)}] top1对={stats['top1_correct']} "
                      f"差点答对={stats['almost_right']} 差得远={stats['far_wrong']}", flush=True)

    total = stats["total"]
    print(f"\n=== 选题统计 ===")
    print(f"总题数 {total}  top1对 {stats['top1_correct']}({100.0*stats['top1_correct']/total:.1f}%)  "
          f"差点答对 {stats['almost_right']}({100.0*stats['almost_right']/total:.1f}%)  "
          f"差得远 {stats['far_wrong']}")
    print(f"-> {OUT_SELECT} ({stats['almost_right']} 题)")


if __name__ == "__main__":
    main()
