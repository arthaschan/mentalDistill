#!/usr/bin/env python3
"""英文牙科"弱教师"组合评估：测 Llama-70B 训练后 3-seed 牙科成绩，与弱教师 Qwen3-32B 对比。

- Qwen2.5-32B 训练后牙科 65.65%（fullEnglish 主实验已知）。
- Llama-70B 训练后 adapter 复用 fullEnglish/03_main_distill/runs/Llama70B_a00_*。
"""
import json
import os
import re
import statistics
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
RUN = os.path.join(ROOT, "runs")
FE_RUNS = os.path.join(ROOT, "..", "fullEnglish", "03_main_distill", "runs")

DENT = re.compile(
    r'\b(tooth|teeth|dental|dentine|dentin|enamel|pulp|molar|premolar|incisor|canine|'
    r'gingiv|periodont|oral|mandib|maxill|caries|occlus|denture|endodont|orthodont|'
    r'amalgam|prosthodont|alveolar|cementum|odonto|crown|root canal|fluoride|saliva|'
    r'palat|buccal|lingual|mucosa|periapical|dentition|bruxism|malocclusion)\b', re.I)

WEAK_TEACHER = {"name": "Qwen3-32B(弱教师)", "acc": 62.76}
FLASH = 70.0
QWEN25_32B_TRAINED = 65.65

os.environ["DISTILL_USE_CHAT_TEMPLATE"] = "1"
os.environ["DISTILL_PROMPT_LANG"] = "en"
sys.path.insert(0, os.path.join(ROOT, "..", "shared"))
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
    out = []
    for s in ["test_medqa", "test_medmcqa", "test_mmlu"]:
        for r in load(os.path.join(DATA, f"{s}.jsonl")):
            if DENT.search(" ".join([str(r.get("Question", "")), str(r.get("Options", ""))])):
                out.append({"Question": r.get("Question", ""), "Options": r.get("Options", ""),
                            "Answer": str(r.get("Answer", "")).strip().upper()})
    return out


def main():
    from transformers import AutoTokenizer
    from peft import PeftModel

    dental = collect_dental()
    print(f"牙科题: {len(dental)}", flush=True)

    model_name = os.path.join(ROOT, "..", "models", "Llama-3.3-70B-Instruct")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = load_base_model(model_name, "4bit", device)

    per = {}
    for s in ["s11", "s42", "s8"]:
        ad = os.path.join(FE_RUNS, f"Llama70B_a00_{s}", "best")
        model = PeftModel.from_pretrained(base, ad)
        model.eval()
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
        per[s] = round(100.0 * c / len(dental), 2)
        print(f"  Llama70B_a00_{s} 牙科: {per[s]}%", flush=True)

    m = statistics.mean(per.values())
    sd = statistics.stdev(per.values())
    print(f"\n=== 英文牙科(980题) 弱教师组合 ===")
    print(f"弱教师 Qwen3-32B 零样本: {WEAK_TEACHER['acc']}%")
    print(f"学生 Qwen2.5-32B 训练后: {QWEN25_32B_TRAINED}% → 超弱教师 {QWEN25_32B_TRAINED - WEAK_TEACHER['acc']:+.2f}pp")
    print(f"学生 Llama-70B 训练后: {m:.2f}±{sd:.2f}% ({per}) → 超弱教师 {m - WEAK_TEACHER['acc']:+.2f}pp")
    print(f"（强教师 flash 牙科 {FLASH}%，学生都追不上）")

    json.dump({"weak_teacher": WEAK_TEACHER, "flash": FLASH,
               "qwen25_32b_trained": QWEN25_32B_TRAINED, "llama70b_per_seed": per,
               "llama70b_mean": round(m, 2), "llama70b_std": round(sd, 2)},
              open(os.path.join(RUN, "eval_results_en_dental_weakteacher.json"), "w"),
              ensure_ascii=False, indent=2)
    print(f"-> {RUN}/eval_results_en_dental_weakteacher.json")


if __name__ == "__main__":
    main()
