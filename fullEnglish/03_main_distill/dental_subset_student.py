#!/usr/bin/env python3
"""评估 32B 学生 (α=0, 3 seed) 在牙科子集上的准确率, 与 DeepSeek 教师同集对比.

牙科识别复用 dental_subset_teacher.py 的关键词正则 (英文牙科实验 DENT).
"""
import json
import os
import re
import statistics

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DENT = re.compile(
    r'\b(tooth|teeth|dental|dentine|dentin|enamel|pulp|molar|premolar|incisor|canine|'
    r'gingiv|periodont|oral|mandib|maxill|caries|occlus|denture|endodont|orthodont|'
    r'amalgam|prosthodont|alveolar|cementum|odonto|crown|root canal|fluoride|saliva|'
    r'palat|buccal|lingual|mucosa|periapical|dentition|bruxism|malocclusion)\b', re.I)

LETTERS = ["A", "B", "C", "D", "E"]
SYSTEM_LINE = ("You are a medical expert. Output exactly one letter "
               "(A, B, C, D, or E) as the answer, with no explanation or spaces.\n")
DATA = "fullEnglish/00_data/out"
LABELS = "fullEnglish/03_main_distill/labels"
RUN = "fullEnglish/03_main_distill/runs"
STUDENT = os.environ.get("BASE_MODEL_32B", "models/Qwen2.5-32B-Instruct")
SEEDS = ["s11", "s42", "s8"]
device = "cuda" if torch.cuda.is_available() else "cpu"


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


def is_dental(r):
    return bool(DENT.search(" ".join([str(r.get("Question", "")), str(r.get("Options", ""))])))


def build_prompt(q, opts):
    return ("<|im_start|>system\n" + SYSTEM_LINE + "<|im_end|>\n"
            "<|im_start|>user\n" + f"Question: {q}\nOptions:\n{opts}\n" + "<|im_end|>\n"
            "<|im_start|>assistant\n")


def extract(text):
    for ch in text.strip().upper():
        if ch in LETTERS:
            return ch
    return ""


# 收集牙科题
dental = []
for s in ["test_medqa", "test_medmcqa", "test_mmlu"]:
    for r in load(f"{DATA}/{s}.jsonl"):
        if is_dental(r):
            dental.append({"uid": r["uid"], "Question": r["Question"],
                           "Options": r["Options"], "Answer": str(r["Answer"]).strip().upper(),
                           "source": r.get("source", "?")})
print(f"牙科题合计: {len(dental)}")

# 老师标签
teacher = {}
for s in ["test_medqa", "test_medmcqa", "test_mmlu"]:
    p = f"{LABELS}/teacher_{s}.jsonl"
    if os.path.exists(p):
        for r in load(p):
            if r.get("uid"):
                teacher[r["uid"]] = r
# 老师牙科准确率
t_c = sum(1 for r in dental if teacher.get(r["uid"]) and
          str(teacher[r["uid"]].get("TeacherAnswer", "")).upper() == str(teacher[r["uid"]].get("OriginalAnswer", "")).upper())
t_acc = round(100 * t_c / len(dental), 2)
print(f"教师 DeepSeek 牙科准确率: {t_acc}% ({t_c}/{len(dental)})")

# 学生 3 seed 评估
student = {}
for seed in SEEDS:
    ad = f"{RUN}/32B_a00_{seed}/best"
    print(f"=== 评估 32B_a00_{seed} 牙科 ===", flush=True)
    model = AutoModelForCausalLM.from_pretrained(STUDENT, torch_dtype=torch.bfloat16,
                                                 device_map=device, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(STUDENT, trust_remote_code=True)
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, ad)
    model.eval()
    correct = 0
    for r in dental:
        inputs = tok(build_prompt(r["Question"], r["Options"]), return_tensors="pt", truncation=True).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=4, do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        gen = tok.decode(out[0][inputs["input_ids"].size(1):], skip_special_tokens=True)
        if extract(gen) == r["Answer"]:
            correct += 1
    student[seed] = round(100 * correct / len(dental), 2)
    print(f"  32B_a00_{seed} 牙科: {student[seed]}% ({correct}/{len(dental)})", flush=True)
    del model
    torch.cuda.empty_cache()

vals = [student[s] for s in SEEDS]
m, sd = statistics.mean(vals), statistics.stdev(vals)
print(f"\n=== 牙科子集 (n={len(dental)}) 学生 vs 教师 ===")
print(f"教师 DeepSeek: {t_acc}%")
print(f"学生 32B α=0: {m}±{sd}% ({student})")
print(f"Δ(学生-教师): {m - t_acc:+.2f}pp  -> {'超越' if m > t_acc else '未超越'}")

out = {"n_dental": len(dental), "teacher_acc": t_acc,
       "student_per_seed": student, "student_mean": round(m, 2), "student_std": round(sd, 2),
       "delta": round(m - t_acc, 2)}
json.dump(out, open(f"{RUN}/dental_subset_result.json", "w"), ensure_ascii=False, indent=2)
print(f"-> {RUN}/dental_subset_result.json")
