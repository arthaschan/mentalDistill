#!/usr/bin/env python3
"""中文牙科子集零样本阶梯 + 4% 组合表（排入队列，GPU 空闲后跑）。

做什么：
1. 用 CMExam 中文牙科子集（15_fulldata_resplit/data/test_dental.jsonl 125 题 + val_dental 125 题 = 250 题）
   测各可部署模型的零样本准确率。
2. 老师锚：DeepSeek-V3 牙科 79.20%（Module 15 README 已知口径，硬编码参考）。
   外加可部署模型本身当"弱老师"。
3. 排 4% 差组合表：老师 − 学生 ≤ 4pp 的对（学生训练后 ~+4.8pp 有望超越）。

提示与中文主线一致：DISTILL_PROMPT_LANG=zh（中文牙科 prompt）+ chat template（Qwen3 自动关 thinking）。
已知参考（README）：DeepSeek 79.20% / 14B 零样本 74.40% / 7B 68.80%；牙科蒸馏增益 ≈ +4.80pp。

注意：中文牙科只有 250 题，零样本每题 ±~6pp 噪声，4% 差组合只做初筛，命中后再用更大集确认。
产出 runs/dental_ladder_zeroshot_cn.json。
"""
import json
import os
import sys

import torch

FE = "fullEnglish/03_main_distill"          # 结果统一写 fullEnglish 的 runs/
CN_DATA = "15_fulldata_resplit/data"
RUN = f"{FE}/runs"
OUT = f"{RUN}/dental_ladder_zeroshot_cn.json"

# 中文牙科测试集：test_dental(125) + val_dental(125) = 250 题（同 split，无重叠）
DENTAL_FILES = [f"{CN_DATA}/test_dental.jsonl", f"{CN_DATA}/val_dental.jsonl"]

# 可部署学生（测零样本阶梯）。quantize='4bit' 走 bitsandbytes NF4（70B 装不下 bf16）。
MODELS = [
    {"name": "Qwen3-32B",     "path": "models/Qwen3-32B",             "quantize": "none"},
    {"name": "Qwen2.5-32B",   "path": "models/Qwen2.5-32B-Instruct",  "quantize": "none"},
    {"name": "Qwen2.5-14B",   "path": "models/Qwen2.5-14B-Instruct",  "quantize": "none"},
    {"name": "Llama-3.3-70B", "path": "models/Llama-3.3-70B-Instruct", "quantize": "4bit"},
]

# 各学生中文牙科蒸馏增益（14B/7B 为 README 实测 +4.80pp；其余用 ~4 估，标注）
GAINS = {"Qwen3-32B": 4.0, "Qwen2.5-32B": 4.0, "Qwen2.5-14B": 4.8, "Llama-3.3-70B": 4.0}

# 已知老师锚（中文牙科，Module 15 README 口径；deepseek-chat = v4-flash）
KNOWN_TEACHERS = {
    "DeepSeek-V3": {"combined": 79.20, "combined_n": 125, "note": "README 口径(125题)"},
}

os.environ["DISTILL_USE_CHAT_TEMPLATE"] = "1"
os.environ["DISTILL_PROMPT_LANG"] = "zh"   # 中文牙科 prompt
sys.path.insert(0, "shared")
from train_choice_head_distill import (  # noqa: E402
    apply_prompt_template, build_mcq_prompt, extract_answer_char, load_base_model,
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


def collect_dental():
    rows = []
    for f in DENTAL_FILES:
        for r in load(f):
            q = r.get("Question", "")
            opts = r.get("Options", "")
            ans = str(r.get("Answer", "")).strip().upper()
            if q and opts and ans in "ABCDE":
                rows.append({"Question": q, "Options": opts, "Answer": ans, "file": os.path.basename(f)})
    return rows


def eval_model(model, tok, dental):
    c = 0
    for r in dental:
        sys_line, user_block = build_mcq_prompt(r["Question"], r["Options"])
        prompt, _ = apply_prompt_template(tok, sys_line, user_block)
        inputs = tok(prompt, return_tensors="pt", truncation=True).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=16, do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        gen = tok.decode(out[0][inputs["input_ids"].size(1):], skip_special_tokens=True)
        if extract_answer_char(gen) == r["Answer"]:
            c += 1
    n = len(dental)
    return {"n": n, "acc": round(100.0 * c / n, 2) if n else None}


def main():
    from transformers import AutoTokenizer

    dental = collect_dental()
    print(f"中文牙科题合计: {len(dental)} 题", flush=True)

    if os.path.exists(OUT):
        results = json.load(open(OUT))
    else:
        results = {"teachers": KNOWN_TEACHERS, "zeroshot": {}, "combos": []}
    results["n_dental"] = len(dental)

    print("=== 老师锚（中文牙科） ===", flush=True)
    for t, v in results.get("teachers", {}).items():
        print(f"  {t:12s} {v['combined']}% {v.get('note', '')}", flush=True)

    # 各模型零样本阶梯（模型级断点续跑）
    for m in MODELS:
        name = m["name"]
        if name in results.get("zeroshot", {}):
            print(f"[SKIP] {name} 已测过", flush=True)
            continue
        print(f"=== 零样本 {name} ({m['quantize']}) ===", flush=True)
        try:
            tok = AutoTokenizer.from_pretrained(m["path"], trust_remote_code=True)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            model = load_base_model(m["path"], m["quantize"], device)
            if m["quantize"] != "4bit":
                model = model.to(device)
            model.eval()
            results.setdefault("zeroshot", {})[name] = eval_model(model, tok, dental)
            del model
            torch.cuda.empty_cache()
            v = results["zeroshot"][name]
            print(f"  {name} 中文牙科零样本 = {v['acc']}% (n={v['n']})", flush=True)
            json.dump(results, open(OUT, "w"), ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"  [ERROR] {name} 评估失败: {e}", flush=True)

    # 4% 组合表
    zeroshot = results.get("zeroshot", {})
    teachers = results.get("teachers", {})
    combos = []
    teacher_names = list(teachers.keys()) + [m["name"] for m in MODELS]
    for tname in teacher_names:
        if tname in teachers:
            t_acc = teachers[tname]["combined"]
        elif tname in zeroshot:
            t_acc = zeroshot[tname]["acc"]
        else:
            continue
        for m in MODELS:
            sname = m["name"]
            if sname == tname:
                continue
            s_acc = zeroshot.get(sname, {}).get("acc")
            if s_acc is None or t_acc is None:
                continue
            gap = round(t_acc - s_acc, 2)
            gain = GAINS.get(sname, 4.0)
            pred = round(s_acc + gain, 2)
            combos.append({
                "teacher": tname, "teacher_acc": t_acc,
                "student": sname, "student_zero": s_acc,
                "gap": gap, "predicted_trained": pred,
                "predicted_delta": round(pred - t_acc, 2),
                "candidate_4pp": 0 < gap <= 4.0,
                "surpass_predicted": pred > t_acc,
            })
    combos.sort(key=lambda c: c["gap"])
    results["combos"] = combos

    print("\n=== 中文牙科 4% 差组合表（≤4pp 有望超越） ===", flush=True)
    print(f"{'老师':14s} {'老师%':>6s} {'学生':13s} {'学生零%':>7s} {'gap':>6s} {'预测训练后%':>8s}  判断", flush=True)
    for c in combos:
        if c["gap"] <= 4.0:
            flag = "★候选" if c["candidate_4pp"] else "  (负gap已超)"
            print(f"{c['teacher']:14s} {c['teacher_acc']:>6.2f} {c['student']:13s} "
                  f"{c['student_zero']:>7.2f} {c['gap']:>+6.2f} {c['predicted_trained']:>8.2f}  {flag}", flush=True)
    print("\n(注: 250 题零样本 ±~6pp 噪声, 4% 差仅初筛, 命中组合需更大集或训练验证)", flush=True)

    json.dump(results, open(OUT, "w"), ensure_ascii=False, indent=2)
    print(f"\n-> {OUT}", flush=True)


if __name__ == "__main__":
    main()
