#!/usr/bin/env python3
"""豆包 API 冒烟测试：验证 key + 模型名 + 返回格式（chat/completions）。"""
import json
import os
import sys
import time

import requests

API_KEY = os.environ.get("DOUBAO_API_KEY", "")
MODEL = "doubao-seed-2-1-pro-260628"
BASE = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"

# 从 screen_input.jsonl 取第一道真实英文医学题
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

for idx, item in enumerate(rows):
    q = item["Question"]
    opts = item.get("Options", {})
    lines = [q]
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
        ], "temperature": 0, "max_tokens": 64},
        timeout=120,
    )
    dt = time.time() - t0
    print(f"\n=== Q{idx+1} (gt={gt}) status={r.status_code} {dt:.1f}s ===")
    if r.status_code != 200:
        print("ERR:", r.text[:500])
        continue
    data = r.json()
    content = data["choices"][0]["message"]["content"]
    print(f"content repr: {content!r}")
    print(f"usage: {data.get('usage')}")
