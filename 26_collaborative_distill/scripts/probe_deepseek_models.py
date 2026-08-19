#!/usr/bin/env python3
"""探测 DeepSeek 官方 API（DEEPSEEK_API_KEY）上 v4-pro 的正确模型名。"""
import json
import os
import sys
import time

import requests

API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
BASE = "https://api.deepseek.com/v1/chat/completions"

CANDIDATES = [
    "deepseek-v4-pro",
    "deepseek-v4-pro-260425",
    "deepseek-chat",
    "deepseek-reasoner",
]

SYS = ("You are a medical expert. Answer the following single-best-answer "
       "multiple-choice question using your professional knowledge. "
       "Output only one capital letter (A/B/C/D/E) as the answer. "
       "Do not output any explanation or extra text.")

PROMPT = ("A 23-year-old pregnant woman at 22 weeks gestation presents with "
          "burning upon urination. Which of the following is the best treatment "
          "for this patient?\n"
          "A. Ampicillin\nB. Ceftriaxone\nC. Ciprofloxacin\nD. Doxycycline\n"
          "E. Nitrofurantoin\n"
          "Output only one capital letter (A/B/C/D/E) as the answer.")


def main():
    if not API_KEY:
        print("[FATAL] DEEPSEEK_API_KEY 未配置", flush=True)
        sys.exit(2)
    print(f"key 长度={len(API_KEY)}", flush=True)
    for name in CANDIDATES:
        t0 = time.time()
        try:
            r = requests.post(
                BASE,
                headers={"Authorization": f"Bearer {API_KEY}",
                         "Content-Type": "application/json"},
                json={"model": name,
                      "messages": [
                          {"role": "system", "content": SYS},
                          {"role": "user", "content": PROMPT},
                      ],
                      "temperature": 0, "max_tokens": 8192},
                timeout=120,
            )
            dt = time.time() - t0
            if r.status_code != 200:
                body = r.text[:160].replace("\n", " ")
                print(f"  {name:24s} -> HTTP {r.status_code} {body}", flush=True)
                continue
            data = r.json()
            content = data["choices"][0]["message"]["content"].strip()
            ans = content.upper()[:1]
            rt = (data.get("usage", {}).get("completion_tokens_details") or {}) \
                .get("reasoning_tokens", 0)
            print(f"  {name:24s} -> OK ans={ans} {dt:.1f}s "
                  f"reasoning={rt}", flush=True)
        except Exception as e:
            dt = time.time() - t0
            print(f"  {name:24s} -> ERR {e} ({dt:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
