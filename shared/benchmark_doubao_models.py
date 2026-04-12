#!/usr/bin/env python3
"""Benchmark multiple Doubao models on dental MCQ for speed + accuracy."""
import json, os, sys, time, requests

API_KEY = os.environ.get("DOUBAO_API_KEY", "")
BASE = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
SYSTEM = "请根据以下临床医学选择题选出正确答案，只需输出一个大写字母（A/B/C/D/E），不要输出任何其他内容。"

def load_questions(path, n=5):
    qs = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            d = json.loads(line)
            q = d["Question"]
            opts = d.get("Options", "")
            if isinstance(opts, dict):
                opt_text = "\n".join(f"{k}. {opts[k]}" for k in "ABCDE" if k in opts)
            else:
                opt_text = str(opts)
            prompt = f"{q}\n{opt_text}\n请只输出一个大写字母（A/B/C/D/E）。"
            gt = str(d.get("Answer", "")).strip().upper()[:1]
            qs.append((prompt, gt))
    return qs

def call_model(model_id, prompt, temp=0.9):
    t0 = time.time()
    r = requests.post(BASE,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json={"model": model_id, "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt}
        ], "temperature": temp, "max_tokens": 16},
        timeout=120)
    dt = time.time() - t0
    if r.status_code != 200:
        return None, dt, f"HTTP {r.status_code}"
    ans = r.json()["choices"][0]["message"]["content"].strip()
    return ans, dt, None

def main():
    data_path = sys.argv[1] if len(sys.argv) > 1 else "03_doubao_choice_head/data/teacher_train.jsonl"
    questions = load_questions(data_path, n=5)

    models = [
        ("ep-20260330210911-6pc2r",    "seed-2.0-pro (reasoning)"),
        ("doubao-1-5-pro-32k-250115",  "1.5-pro-32k (chat)"),
        ("doubao-seed-1-6-flash-250828","seed-1.6-flash"),
        ("doubao-seed-2-0-lite-260215","seed-2.0-lite"),
        ("doubao-seed-2-0-mini-260215","seed-2.0-mini"),
    ]

    for model_id, name in models:
        print(f"\n=== {name} ({model_id}) ===", flush=True)
        total_time = 0
        correct = 0
        errors = 0
        for qi, (q, gt) in enumerate(questions):
            ans, dt, err = call_model(model_id, q)
            total_time += dt
            if err:
                errors += 1
                print(f"  Q{qi+1}: ERR {err} {dt:.1f}s", flush=True)
            else:
                letter = ans.upper()[:1]
                ok = "Y" if letter == gt else "N"
                if ok == "Y":
                    correct += 1
                print(f"  Q{qi+1}: {ans[:20]:20s} (gt={gt}) {ok} {dt:.1f}s", flush=True)
        avg = total_time / max(len(questions), 1)
        print(f"  => avg={avg:.1f}s/q  correct={correct}/{len(questions)}  errors={errors}", flush=True)

if __name__ == "__main__":
    main()
