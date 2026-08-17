#!/usr/bin/env python3
"""牙科零样本阶梯 + 4% 组合表（PED 跑完后跑这一轮）。

做什么：
1. 用 DENT 正则从 test_medqa/medmcqa/mmlu 筛出牙科题（980 题，与 dental_subset_*.py 同口径）。
2. 从已有教师标签（labels/teacher_test_*.jsonl）算各老师牙科准确率：
   flash / dsv4pro / glm52 / doubao（glm52、doubao 无 mmlu 标签，口径 913）。
3. 测各可部署模型在牙科子集上的零样本准确率（阶梯）：
   Qwen3-32B(非思考) / Qwen2.5-32B / Qwen2.5-14B / Llama-3.3-70B(4bit)。
4. 排"4% 差组合表"：老师零样本 − 学生零样本 ≤ 4pp 的对（学生训练后 ~+4pp 有望超越）。

提示与主线 eval 完全一致：DISTILL_USE_CHAT_TEMPLATE=1（Qwen3 自动 enable_thinking=False）。
产出 runs/dental_ladder_zeroshot.json（含阶梯 + 教师 + 组合表）。

注意：本脚本吃 GPU，等 PED 训练结束后再跑（H100 95GB，一次只载一个模型）。
"""
import json
import os
import re
import sys

import torch

FE = "fullEnglish/03_main_distill"
DATA = "fullEnglish/00_data/out"
LABELS = f"{FE}/labels"
RUN = f"{FE}/runs"
OUT = f"{RUN}/dental_ladder_zeroshot.json"

# 牙科识别正则（复用 dental_subset_teacher.py）
DENT = re.compile(
    r'\b(tooth|teeth|dental|dentine|dentin|enamel|pulp|molar|premolar|incisor|canine|'
    r'gingiv|periodont|oral|mandib|maxill|caries|occlus|denture|endodont|orthodont|'
    r'amalgam|prosthodont|alveolar|cementum|odonto|crown|root canal|fluoride|saliva|'
    r'palat|buccal|lingual|mucosa|periapical|dentition|bruxism|malocclusion)\b', re.I)

SOURCES = ["test_medqa", "test_medmcqa", "test_mmlu"]

# 可部署学生（测零样本阶梯）。quantize='4bit' 走 bitsandbytes NF4（70B 装不下 bf16）。
MODELS = [
    {"name": "Qwen3-32B",    "path": "models/Qwen3-32B",            "quantize": "none"},
    {"name": "Qwen2.5-32B",  "path": "models/Qwen2.5-32B-Instruct", "quantize": "none"},
    {"name": "Qwen2.5-14B",  "path": "models/Qwen2.5-14B-Instruct", "quantize": "none"},
    {"name": "Llama-3.3-70B", "path": "models/Llama-3.3-70B-Instruct", "quantize": "4bit"},
]

# 各学生已知蒸馏增益（全量 8293 口径；14B 用 ~4 估）
GAINS = {"Qwen3-32B": 3.75, "Qwen2.5-32B": 4.26, "Llama-3.3-70B": 3.84, "Qwen2.5-14B": 4.0}

# 教师标签文件（teacher = 锚，其牙科零样本成绩从 labels 算）
TEACHER_LABELS = {
    "flash":    {"test_medqa": "teacher_test_medqa.jsonl",
                 "test_medmcqa": "teacher_test_medmcqa.jsonl",
                 "test_mmlu": "teacher_test_mmlu.jsonl"},
    "dsv4pro":  {"test_medqa": "teacher_test_medqa_dsv4pro.jsonl",
                 "test_medmcqa": "teacher_test_medmcqa_dsv4pro.jsonl",
                 "test_mmlu": "teacher_test_mmlu_dsv4pro.jsonl"},
    "glm52":    {"test_medqa": "teacher_test_medqa_glm52.jsonl",
                 "test_medmcqa": "teacher_test_medmcqa_glm52.jsonl"},
    "doubao":   {"test_medqa": "teacher_test_medqa_doubao.jsonl",
                 "test_medmcqa": "teacher_test_medmcqa_doubao.jsonl"},
}

os.environ["DISTILL_USE_CHAT_TEMPLATE"] = "1"
os.environ["DISTILL_PROMPT_LANG"] = "en"
sys.path.insert(0, "shared")
from train_choice_head_distill import (  # noqa: E402
    apply_prompt_template, build_mcq_prompt, extract_answer_char, load_base_model,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load(path):
    rows = []
    if not os.path.exists(path):
        return rows
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


def collect_dental():
    """筛牙科题，返回 list[dict]，带 source。"""
    dental = []
    for s in SOURCES:
        for r in load(f"{DATA}/{s}.jsonl"):
            if is_dental(r):
                dental.append({"uid": r.get("uid"),
                               "Question": r.get("Question", ""),
                               "Options": r.get("Options", ""),
                               "Answer": str(r.get("Answer", "")).strip().upper(),
                               "source": s})
    return dental


def teacher_accs(dental):
    """从标签算各老师牙科准确率（per-source + combined）。"""
    # 按 uid 建索引：uid -> {teacher: answer}
    label_idx = {}
    for tname, fmap in TEACHER_LABELS.items():
        for s, fname in fmap.items():
            for r in load(f"{LABELS}/{fname}"):
                uid = r.get("uid")
                if not uid:
                    continue
                label_idx.setdefault(uid, {})[tname] = r

    result = {}
    for tname in TEACHER_LABELS:
        per = {}
        tot_c = tot_n = 0
        for s in SOURCES:
            rows = [r for r in dental if r["source"] == s]
            n = len(rows)
            if n == 0:
                per[s] = None
                continue
            c = 0
            covered = 0
            for r in rows:
                tr = label_idx.get(r["uid"], {}).get(tname)
                if tr is None:
                    continue
                covered += 1
                gt = str(tr.get("OriginalAnswer") or tr.get("Answer", "")).strip().upper()
                ta = str(tr.get("TeacherAnswer") or tr.get("Answer", "")).strip().upper()
                if gt in "ABCDE" and ta in "ABCDE" and ta == gt:
                    c += 1
            per[s] = {"n": n, "covered": covered,
                      "acc": round(100 * c / covered, 2) if covered else None}
            tot_c += c
            tot_n += covered
        result[tname] = {"per_source": per,
                         "combined_n": tot_n,
                         "combined": round(100 * tot_c / tot_n, 2) if tot_n else None,
                         # 覆盖不足 → 牙科成绩不可靠（pro/glm/doubao 只有 600 筛选子集标签残留）
                         "low_coverage": tot_n < 500}
    return result


def eval_model(model, tok, dental):
    """测一个模型在牙科子集上的零样本准确率（per-source + combined）。"""
    per = {}
    tot_c = tot_n = 0
    for s in SOURCES:
        rows = [r for r in dental if r["source"] == s]
        c = 0
        for r in rows:
            sys_line, user_block = build_mcq_prompt(r["Question"], r["Options"])
            prompt, _ = apply_prompt_template(tok, sys_line, user_block)
            inputs = tok(prompt, return_tensors="pt", truncation=True).to(device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=16, do_sample=False,
                                     pad_token_id=tok.pad_token_id or tok.eos_token_id)
            gen = tok.decode(out[0][inputs["input_ids"].size(1):], skip_special_tokens=True)
            if extract_answer_char(gen) == r["Answer"]:
                c += 1
        n = len(rows)
        per[s] = {"n": n, "acc": round(100 * c / n, 2) if n else None}
        tot_c += c
        tot_n += n
    return {"per_source": per, "combined_n": tot_n,
            "combined": round(100 * tot_c / tot_n, 2) if tot_n else None}


def main():
    from transformers import AutoTokenizer

    dental = collect_dental()
    print(f"牙科题合计: {len(dental)} 题", flush=True)

    # 已有结果续跑（模型级断点续跑）
    if os.path.exists(OUT):
        results = json.load(open(OUT))
    else:
        results = {"teachers": {}, "zeroshot": {}, "combos": []}
    results["n_dental"] = len(dental)

    # ① 教师牙科成绩（纯 CPU，秒出）
    if not results.get("teachers"):
        results["teachers"] = teacher_accs(dental)
        print("=== 教师牙科准确率（标签口径） ===", flush=True)
        for t, v in results["teachers"].items():
            cov = "  ⚠低覆盖,不可靠" if v.get("low_coverage") else ""
            print(f"  {t:9s} combined={v['combined']}% (n={v['combined_n']}){cov}", flush=True)
        json.dump(results, open(OUT, "w"), ensure_ascii=False, indent=2)

    # ② 各模型零样本阶梯
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
            model.eval()
            results.setdefault("zeroshot", {})[name] = eval_model(model, tok, dental)
            del model
            torch.cuda.empty_cache()
            v = results["zeroshot"][name]
            print(f"  {name} 牙科零样本 combined={v['combined']}% (n={v['combined_n']})", flush=True)
            json.dump(results, open(OUT, "w"), ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"  [ERROR] {name} 评估失败: {e}", flush=True)

    # ③ 4% 组合表
    zeroshot = results.get("zeroshot", {})
    teachers = results.get("teachers", {})
    # 老师 = 4 个 API 锚 + 可部署模型本身（弱老师）
    combos = []
    teacher_names = list(teachers.keys()) + [m["name"] for m in MODELS]
    for tname in teacher_names:
        t_low = False
        if tname in teachers:
            t_acc = teachers[tname]["combined"]
            t_low = teachers[tname].get("low_coverage", False)
        elif tname in zeroshot:
            t_acc = zeroshot[tname]["combined"]
        else:
            continue
        for m in MODELS:
            sname = m["name"]
            if sname == tname:
                continue
            s_acc = zeroshot.get(sname, {}).get("combined")
            if s_acc is None or t_acc is None:
                continue
            gap = round(t_acc - s_acc, 2)
            gain = GAINS.get(sname, 4.0)
            pred = round(s_acc + gain, 2)   # 训练后预测
            combos.append({
                "teacher": tname, "teacher_acc": t_acc,
                "teacher_low_coverage": t_low,
                "student": sname, "student_zero": s_acc,
                "gap": gap, "predicted_trained": pred,
                "predicted_delta": round(pred - t_acc, 2),
                "candidate_4pp": 0 < gap <= 4.0,
                "surpass_predicted": pred > t_acc,
            })
    combos.sort(key=lambda c: c["gap"])
    results["combos"] = combos

    print("\n=== 4% 差组合表（gap = 老师零样本 − 学生零样本，≤4pp 有望超越） ===", flush=True)
    print(f"{'老师':12s} {'老师%':>6s} {'学生':13s} {'学生零%':>7s} {'gap':>6s} {'预测训练后%':>8s}  判断", flush=True)
    for c in combos:
        if c["gap"] <= 4.0:
            flag = "★候选" if c["candidate_4pp"] else "  (负gap已超)"
            low = " [老师低覆盖]" if c.get("teacher_low_coverage") else ""
            print(f"{c['teacher']:12s} {c['teacher_acc']:>6.2f} {c['student']:13s} "
                  f"{c['student_zero']:>7.2f} {c['gap']:>+6.2f} {c['predicted_trained']:>8.2f}  {flag}{low}", flush=True)
    print("\n(注: glm52/doubao 缺 mmlu 牙科标签, 口径=913 题; 其余 980 题)", flush=True)
    print("(注: 标 [老师低覆盖] 的教师牙科成绩来自 600 筛选子集残留, n<500, 不可靠, 需补 API 全量)", flush=True)

    json.dump(results, open(OUT, "w"), ensure_ascii=False, indent=2)
    print(f"\n-> {OUT}", flush=True)


if __name__ == "__main__":
    main()
