#!/usr/bin/env python3
"""对比 seed-2.1-pro vs seed-2.1-turbo：同 3 道医学题，测延迟 + 正确率。"""
import json
import os
import time

import requests

API_KEY = os.environ.get("DOUBAO_API_KEY", "")
BASE = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
MODELS = [
    "doubao-seed-2-1-pro-260628",
    "doubao-seed-2-1-turbo-260628",
]

rows = []
with open("fullEnglish/00_data/out/screen_input.jsonl") as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))
        if len(rows) >= 3:
            break

sys_prompt = ("You are a medical expert. Answer the following single-best-answer "
              "multiple-choice question using your professional knowledge. "
              "Output only one capital letter (A/B/C/D/E) as the answer. "
              "Do not output any explanation or extra text.")
trailing = "Output only one capital letter (A/B/C/D/E) as the answer. Do not output any explanation or extra text."

prompts = []
for item in rows:
    lines = [item["Question"]]
    opts = item.get("Options", {})
    if isinstance(opts, dict):
        for k in ["A", "B", "C", "D", "E"]:
            if k in opts:
                lines.append(f"{k}. {str(opts[k]).strip()}")
    else:
        lines.append(str(opts).strip())
    lines.append(trailing)
    prompts.append((item["Question"], "\n".join(lines), str(item.get("Answer", "")).strip().upper()))

for model in MODELS:
    print(f"\n===== {model} =====")
    times = []
    correct = 0
    for qi, (_, prompt, gt) in enumerate(prompts):
        t0 = time.time()
        r = requests.post(
            BASE,
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            json={"model": model, "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt},
            ], "temperature": 0, "max_tokens": 64},
            timeout=180,
        )
        dt = time.time() - t0
        if r.status_code != 200:
            print(f"  Q{qi+1}: HTTP {r.status_code} {r.text[:200]} ({dt:.1f}s)")
            continue
        data = r.json()
        ans = data["choices"][0]["message"]["content"].strip().upper()[:1]
        rt = data["usage"]["completion_tokens_details"].get("reasoning_tokens", 0)
        ok = "Y" if ans == gt else "N"
        correct += (ans == gt)
        times.append(dt)
        print(f"  Q{qi+1}: ans={ans} gt={gt} {ok}  {dt:.1f}s  reasoning_tokens={rt}")
    print(f"  => correct={correct}/3  avg={sum(times)/len(times):.1f}s/q  max={max(times):.1f}s")
