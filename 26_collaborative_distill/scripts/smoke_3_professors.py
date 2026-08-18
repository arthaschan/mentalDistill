#!/usr/bin/env python3
"""冒烟测试：DOUBAO_API_KEY 能否调 3 大教授（dsv4pro / glm52 / doubao-turbo）。

对测试集前 3 道真实英文医学题各答一次，验证：端点鉴权、模型 id 有效、能输出答案字母。
不做训练、不写标签，纯验证。
"""
import json
import os
import sys
import time

import requests

API_KEY = os.environ.get("DOUBAO_API_KEY", "")
BASE = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"

MODELS = [
    ("deepseek-v4-pro-260425",        "DeepSeek-V4-Pro"),
    ("glm-5-2-260617",                "GLM-5.2"),
    ("doubao-seed-2-1-turbo-260628",  "豆包 seed-2.1-turbo"),
]

SYS = ("You are a medical expert. Answer the following single-best-answer "
       "multiple-choice question using your professional knowledge. "
       "Output only one capital letter (A/B/C/D/E) as the answer. "
       "Do not output any explanation or extra text.")
TRAILING = ("Output only one capital letter (A/B/C/D/E) as the answer. "
            "Do not output any explanation or extra text.")


def load_questions(path, n=3):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            opts = d.get("Options", "")
            if isinstance(opts, dict):
                opt_text = "\n".join(f"{k}. {opts[k]}" for k in "ABCDE" if k in opts)
            else:
                opt_text = str(opts)
            prompt = f"{d['Question']}\n{opt_text}\n{TRAILING}"
            rows.append((prompt, str(d.get("Answer", "")).strip().upper()))
            if len(rows) >= n:
                break
    return rows


def main():
    if not API_KEY:
        print("[FATAL] DOUBAO_API_KEY 未配置", flush=True)
        sys.exit(2)
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "fullEnglish/00_data/out/test_medqa.jsonl"
    qs = load_questions(path)
    print(f"key 长度={len(API_KEY)} 题目数={len(qs)}", flush=True)

    ok_all = True
    for model_id, name in MODELS:
        correct = 0
        errors = 0
        print(f"\n=== {name} ({model_id}) ===", flush=True)
        for qi, (prompt, gt) in enumerate(qs):
            t0 = time.time()
            try:
                r = requests.post(
                    BASE,
                    headers={"Authorization": f"Bearer {API_KEY}",
                             "Content-Type": "application/json"},
                    json={"model": model_id,
                          "messages": [
                              {"role": "system", "content": SYS},
                              {"role": "user", "content": prompt},
                          ],
                          "temperature": 0, "max_tokens": 8192},
                    timeout=300,
                )
                dt = time.time() - t0
                if r.status_code != 200:
                    print(f"  Q{qi+1}: HTTP {r.status_code} {r.text[:200]} ({dt:.1f}s)",
                          flush=True)
                    errors += 1
                    ok_all = False
                    continue
                data = r.json()
                ans = data["choices"][0]["message"]["content"].strip().upper()[:1]
                rt = (data.get("usage", {}).get("completion_tokens_details") or {}) \
                    .get("reasoning_tokens", 0)
                is_ok = "Y" if ans == gt else "N"
                correct += (ans == gt)
                print(f"  Q{qi+1}: ans={ans} gt={gt} {is_ok} {dt:.1f}s "
                      f"reasoning_tokens={rt}", flush=True)
            except Exception as e:
                print(f"  Q{qi+1}: 异常 {e}", flush=True)
                errors += 1
                ok_all = False
        print(f"  => correct={correct}/{len(qs)} errors={errors}", flush=True)

    print(f"\n[RESULT] {'ALL OK' if ok_all else 'HAS ERRORS'}", flush=True)


if __name__ == "__main__":
    main()
