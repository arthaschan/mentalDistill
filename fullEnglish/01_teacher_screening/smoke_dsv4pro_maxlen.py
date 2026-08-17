#!/usr/bin/env python3
"""重测 deepseek-v4-pro：max_tokens 放大后能否正常输出答案。"""
import json
import os
import time

import requests

API_KEY = os.environ.get("DOUBAO_API_KEY", "")
BASE = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
MODEL = "deepseek-v4-pro-260425"

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
    prompt = "\n".join(lines)
    gt = str(item.get("Answer", "")).strip().upper()

    t0 = time.time()
    r = requests.post(
        BASE,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json={"model": MODEL, "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ], "temperature": 0, "max_tokens": 8192},
        timeout=300,
    )
    dt = time.time() - t0
    if r.status_code != 200:
        print(f"HTTP {r.status_code} {r.text[:200]} ({dt:.1f}s)", flush=True)
        continue
    data = r.json()
    content = data["choices"][0]["message"]["content"].strip()
    ans = content.upper()[:1]
    usage = data.get("usage", {})
    rt = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
    ct = usage.get("completion_tokens", 0)
    ok = "Y" if ans == gt else "N"
    print(f"gt={gt} ans={ans!r} {ok}  {dt:.1f}s  reasoning_tokens={rt} completion={ct}", flush=True)
    print(f"  原始输出前 120 字: {content[:120]!r}", flush=True)
