#!/usr/bin/env python3
"""评估"无印度"重训后的学生 + 弱教师 Qwen3-32B，在无印度测试集(全科+牙科)上。

学生 = Qwen2.5-32B + adapter(32B_noindia_a00_s42)
老师 = Qwen3-32B 零样本（弱教师组合）
口径：test_no_india.jsonl(4110) + test_no_india_dental.jsonl(181)
"""
import os
import sys

os.environ["DISTILL_USE_CHAT_TEMPLATE"] = "1"
os.environ["DISTILL_PROMPT_LANG"] = "en"
sys.path.insert(0, "shared")

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from peft import PeftModel  # noqa: E402
from train_choice_head_distill import (  # noqa: E402
    apply_prompt_template, build_mcq_prompt, extract_answer_char, load_base_model,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def eval_file(model, tok, path):
    samples = []
    import json
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("Question") and r.get("Options") and r.get("Answer"):
            samples.append((r["Question"], r["Options"], str(r["Answer"]).strip().upper()))
    correct = 0
    model.eval()
    with torch.no_grad():
        for q, opts, ans in samples:
            sys_line, user_block = build_mcq_prompt(q, opts)
            prompt, _ = apply_prompt_template(tok, sys_line, user_block)
            inputs = tok(prompt, return_tensors="pt", truncation=True).to(device)
            out = model.generate(**inputs, max_new_tokens=4, do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
            gen = tok.decode(out[0][inputs["input_ids"].size(1):], skip_special_tokens=True)
            if extract_answer_char(gen) == ans:
                correct += 1
    return round(100.0 * correct / len(samples), 2) if samples else None


def main():
    tests = {
        "全科(无印度 4110)": "26_collaborative_distill/data/test_no_india.jsonl",
        "牙科(无印度 181)": "26_collaborative_distill/data/test_no_india_dental.jsonl",
    }

    # 1) 学生 Qwen2.5-32B + adapter
    stu_path = "models/Qwen2.5-32B-Instruct"
    adapter = "26_collaborative_distill/runs/32B_noindia_a00_s42"
    print("[学生] 加载 Qwen2.5-32B + adapter ...", flush=True)
    tok = AutoTokenizer.from_pretrained(stu_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = load_base_model(stu_path, "none", device)
    base = base.to(device)
    stu = PeftModel.from_pretrained(base, adapter)
    stu.eval()
    stu_results = {}
    for name, p in tests.items():
        acc = eval_file(stu, tok, p)
        stu_results[name] = acc
        print(f"  学生 {name}: {acc}%", flush=True)
    del stu, base
    torch.cuda.empty_cache()

    # 2) 老师 Qwen3-32B 零样本
    tea_path = "models/Qwen3-32B"
    print("[老师] 加载 Qwen3-32B 零样本 ...", flush=True)
    tok2 = AutoTokenizer.from_pretrained(tea_path, trust_remote_code=True)
    if tok2.pad_token is None:
        tok2.pad_token = tok2.eos_token
    tea = load_base_model(tea_path, "none", device)
    tea = tea.to(device)
    tea.eval()
    tea_results = {}
    for name, p in tests.items():
        acc = eval_file(tea, tok2, p)
        tea_results[name] = acc
        print(f"  老师 {name}: {acc}%", flush=True)

    print("\n=== 无印度：学生 vs 弱教师 Qwen3-32B ===", flush=True)
    for name in tests:
        s = stu_results[name]
        t = tea_results[name]
        d = round(s - t, 2)
        flag = "超" if s > t else "不超"
        print(f"  {name}: 学生 {s}% vs 老师 {t}%  Δ{s:+}  {flag}", flush=True)


if __name__ == "__main__":
    main()
