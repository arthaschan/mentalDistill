#!/usr/bin/env python3
"""冒烟测试：ark key 能否调 GLM 系列 + deepseek-v4-pro（验鉴权/推理/延迟）。"""
import json
import os
import time

import requests

API_KEY = os.environ.get("DOUBAO_API_KEY", "")
BASE = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
MODELS = [
    "glm-5-2-260617",
    "glm-4-7-251222",
    "glm-4-5-air-20250728",
    "deepseek-v4-pro-260425",
]

rows = []
with open("fullEnglish/00_data/out/screen_input.jsonl") as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))
        if len(rows) >= 2:
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
    prompts.append(("\n".join(lines), str(item.get("Answer", "")).strip().upper()))

for model in MODELS:
    print(f"\n===== {model} =====", flush=True)
    for qi, (prompt, gt) in enumerate(prompts):
        t0 = time.time()
        try:
            r = requests.post(
                BASE,
                headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                json={"model": model, "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt},
                ], "temperature": 0, "max_tokens": 64},
                timeout=180,
            )
        except Exception as e:
            print(f"  Q{qi+1}: 异常 {e}", flush=True)
            continue
        dt = time.time() - t0
        if r.status_code != 200:
            print(f"  Q{qi+1}: HTTP {r.status_code} {r.text[:200]} ({dt:.1f}s)", flush=True)
            continue
        data = r.json()
        ans = data["choices"][0]["message"]["content"].strip().upper()[:1]
        usage = data.get("usage", {})
        rt = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
        ok = "Y" if ans == gt else "N"
        print(f"  Q{qi+1}: ans={ans} gt={gt} {ok}  {dt:.1f}s  reasoning_tokens={rt}", flush=True)
