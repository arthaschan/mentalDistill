#!/usr/bin/env python3
"""PED 手段① 第一步：算 Qwen3-32B 在训练集上的零样本 ABCDE 分布，筛出"差点答对"的题。

- enable_thinking=False（非思考，与训练/评估一致）。
- 对每题取最后一个 token 的 logits，在 ABCDE 上 softmax，得到 top1/top2。
- "差点答对" = 学生 top1 答错、但正确答案是 top2。
- 产出：① 全量 logprobs（qwen3_train_logprobs.jsonl，供后续 top-k 分析）
         ② 选题训练集（train_head_almostright.jsonl，只含"差点答对"的题）
"""
import json
import math
import os
import sys

import torch

FE = "fullEnglish/03_main_distill"
DATA = "fullEnglish/00_data/out"
STUDENT = "models/Qwen3-32B"
TRAIN = f"{DATA}/train.jsonl"
OUT_LOGPROBS = f"{FE}/data/qwen3_train_logprobs.jsonl"
OUT_SELECT = f"{FE}/data/train_head_almostright.jsonl"
LETTERS = ["A", "B", "C", "D", "E"]

os.environ["DISTILL_PROMPT_LANG"] = "en"
sys.path.insert(0, "shared")
from train_choice_head_distill import build_mcq_prompt  # noqa: E402


def get_option_token_ids(tokenizer):
    ids = {}
    for letter in LETTERS:
        direct = tokenizer.encode(letter, add_special_tokens=False)
        if len(direct) == 1:
            ids[letter] = direct[0]
        else:
            ids[letter] = tokenizer.encode(f" {letter}", add_special_tokens=False)[-1]
    return ids


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(STUDENT, trust_remote_code=True)
    print(f"加载 {STUDENT} (bf16, enable_thinking=False) ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        STUDENT, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="cuda:0",
    )
    model.eval()
    option_ids = get_option_token_ids(tok)
    print(f"option token ids: {option_ids}", flush=True)

    rows = [json.loads(l) for l in open(TRAIN) if l.strip()]
    print(f"训练集 {len(rows)} 题", flush=True)

    os.makedirs(f"{FE}/data", exist_ok=True)
    stats = {"total": 0, "top1_correct": 0, "almost_right": 0, "far_wrong": 0}
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
            top1 = ranked[0]
            top2 = ranked[1]

            stats["total"] += 1
            if top1 == gt:
                stats["top1_correct"] += 1
            elif top2 == gt:
                stats["almost_right"] += 1
            else:
                stats["far_wrong"] += 1

            # ① 全量 logprobs
            rec = dict(item)
            rec["TeacherDist"] = dist
            rec["TeacherAnswer"] = top1
            rec["OriginalAnswer"] = gt
            rec["Top2"] = top2
            wl.write(json.dumps(rec, ensure_ascii=False) + "\n")

            # ② 选题：只保留"差点答对"的题（top1 错、top2 对）
            if top1 != gt and top2 == gt:
                sel = dict(item)
                sel["Answer"] = gt  # GT 答案，α=0 训练用
                ws.write(json.dumps(sel, ensure_ascii=False) + "\n")

            if (i + 1) % 2000 == 0:
                print(f"  [{i+1}/{len(rows)}] top1对={stats['top1_correct']} "
                      f"差点答对={stats['almost_right']} 差得远={stats['far_wrong']}", flush=True)

    total = stats["total"]
    print(f"\n=== 选题统计 ===")
    print(f"总题数: {total}")
    print(f"top1 已答对: {stats['top1_correct']} ({100.0*stats['top1_correct']/total:.1f}%)")
    print(f"差点答对(top2): {stats['almost_right']} ({100.0*stats['almost_right']/total:.1f}%)")
    print(f"差得远: {stats['far_wrong']} ({100.0*stats['far_wrong']/total:.1f}%)")
    print(f"-> 选题训练集 {OUT_SELECT} ({stats['almost_right']} 题)")
    print(f"-> 全量 logprobs {OUT_LOGPROBS}")


if __name__ == "__main__":
    main()
