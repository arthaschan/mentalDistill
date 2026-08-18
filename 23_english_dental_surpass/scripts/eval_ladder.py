#!/usr/bin/env python3
"""英文牙科零样本阶梯：测 4 个可部署模型在牙科子集(980题)的零样本准确率。

弱教师 = Qwen3-32B（其零样本即弱教师锚）；强教师 flash = 70.0%（已知）。
"""
import json
import os
import re
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
RUN = os.path.join(ROOT, "runs")
OUT = os.path.join(RUN, "dental_ladder_zeroshot.json")

DENT = re.compile(
    r'\b(tooth|teeth|dental|dentine|dentin|enamel|pulp|molar|premolar|incisor|canine|'
    r'gingiv|periodont|oral|mandib|maxill|caries|occlus|denture|endodont|orthodont|'
    r'amalgam|prosthodont|alveolar|cementum|odonto|crown|root canal|fluoride|saliva|'
    r'palat|buccal|lingual|mucosa|periapical|dentition|bruxism|malocclusion)\b', re.I)

MODELS = [
    {"name": "Qwen3-32B",     "path": "Qwen3-32B",             "quantize": "none"},
    {"name": "Qwen2.5-32B",   "path": "Qwen2.5-32B-Instruct",  "quantize": "none"},
    {"name": "Qwen2.5-14B",   "path": "Qwen2.5-14B-Instruct",  "quantize": "none"},
    {"name": "Llama-3.3-70B", "path": "Llama-3.3-70B-Instruct", "quantize": "4bit"},
]
FLASH_TEACHER = 70.0

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

    dental = collect_dental()
    print(f"牙科题: {len(dental)}", flush=True)
    results = {"flash_teacher": FLASH_TEACHER, "n_dental": len(dental), "zeroshot": {}}
    if os.path.exists(OUT):
        results = json.load(open(OUT))

    for m in MODELS:
        name = m["name"]
        if name in results["zeroshot"]:
            print(f"[SKIP] {name}", flush=True)
            continue
        path = os.path.join(ROOT, "..", "models", m["path"])
        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = load_base_model(path, m["quantize"], device)
        if m["quantize"] != "4bit":
            model = model.to(device)
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
        acc = round(100.0 * c / len(dental), 2)
        results["zeroshot"][name] = acc
        print(f"  {name} 零样本 = {acc}%", flush=True)
        json.dump(results, open(OUT, "w"), ensure_ascii=False, indent=2)
        del model
        torch.cuda.empty_cache()

    print(f"\n=== 英文牙科零样本阶梯(980题) ===")
    print(f"强教师 flash: {FLASH_TEACHER}%")
    for k, v in results["zeroshot"].items():
        print(f"  {k}: {v}%")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
