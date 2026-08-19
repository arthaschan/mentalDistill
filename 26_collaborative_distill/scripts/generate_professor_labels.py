#!/usr/bin/env python3
"""并发生成单个教授的 MCQ 标签（答案字母），支持断点续跑。

比 shared/generate_teacher_labels_api.py 多了线程并发（reasoning 模型慢，必须并发）。
输出 JSONL：原始字段 + TeacherAnswer + TeacherRaw + OriginalAnswer。

用法:
    python generate_professor_labels.py \
        --candidate configs/professor_dsv4pro.json \
        --dataset fullEnglish/00_data/out/test_medqa.jsonl \
        --output data/labels/dsv4pro_test_medqa.jsonl \
        --system-prompt prompts/system_mcq_en.txt \
        --trailing prompts/trailing_mcq_en.txt \
        --workers 8 --max-tokens 8192 --timeout 300 --resume
"""
import argparse
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
from pathlib import Path

import requests

ANSWER_RE = re.compile(r"\b([A-E])\b")
OPTION_LETTERS = ["A", "B", "C", "D", "E"]


def resolve_api_key(candidate):
    direct = str(candidate.get("api_key", "")).strip()
    if direct:
        return direct
    env = str(candidate.get("api_key_env", "")).strip()
    if env.startswith("sk-") or env.startswith("eyJ"):
        return env
    if re.fullmatch(r"[A-Z_][A-Z0-9_]*", env):
        return os.getenv(env, "").strip()
    return env


def resolve_base_url(candidate):
    base = str(candidate.get("base_url") or candidate.get("api_base") or "").strip()
    if not base:
        raise RuntimeError("missing base_url/api_base")
    if "chat/completions" not in base:
        base = base.rstrip("/") + "/chat/completions"
    return base


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_question_text(item, trailing):
    q = str(item.get("Question") or item.get("question") or "").strip()
    options = item.get("Options") or item.get("options") or {}
    lines = [q]
    if isinstance(options, dict):
        for k in OPTION_LETTERS:
            if k in options:
                lines.append(f"{k}. {str(options[k]).strip()}")
    else:
        opt = str(options).strip()
        if opt:
            lines.append(opt)
    lines.append(trailing)
    return "\n".join(lines)


def sample_key(item):
    q = str(item.get("Question") or item.get("question") or "").strip()
    options = item.get("Options") or item.get("options") or {}
    if isinstance(options, dict):
        opt_text = "\n".join(f"{k}:{str(options.get(k, ''))}" for k in OPTION_LETTERS)
    else:
        opt_text = str(options).strip()
    return hashlib.sha1(f"{q}\n{opt_text}".encode("utf-8")).hexdigest()


def extract_answer_letter(text):
    if not text:
        return ""
    t = text.strip().upper()
    if len(t) == 1 and t in OPTION_LETTERS:
        return t
    m = ANSWER_RE.search(t)
    return m.group(1) if m else ""


def call_api(candidate, system_prompt, user_prompt, timeout, max_tokens,
             base_backoff, backoff_mult, cooldown, jitter):
    api_key = resolve_api_key(candidate)
    if not api_key:
        raise RuntimeError("missing api key")
    base_url = resolve_base_url(candidate)
    headers = {"Authorization": f"Bearer {api_key}",
               "Content-Type": "application/json"}
    payload = {"model": candidate["model"],
               "messages": [
                   {"role": "system", "content": system_prompt},
                   {"role": "user", "content": user_prompt},
               ],
               "temperature": 0, "max_tokens": max_tokens}

    last_err = None
    for attempt in range(5):
        try:
            resp = requests.post(base_url, headers=headers, json=payload, timeout=timeout)
            if resp.status_code >= 400:
                text = (resp.text or "").strip().replace("\n", " ")[:240]
                raise RuntimeError(f"http_{resp.status_code}: {text}")
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            last_err = e
            if attempt < 4:
                msg = str(e).lower()
                if "429" in msg or "rate limit" in msg or "too many requests" in msg:
                    time.sleep(cooldown + random.uniform(0, jitter))
                else:
                    time.sleep(base_backoff * (backoff_mult ** attempt) +
                               random.uniform(0, jitter))
                continue
            raise last_err
    raise last_err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--system-prompt", required=True)
    ap.add_argument("--trailing", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--request-interval", type=float, default=0.0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    candidate = json.load(open(args.candidate, encoding="utf-8"))
    system_prompt = open(args.system_prompt, encoding="utf-8").read().strip()
    trailing = open(args.trailing, encoding="utf-8").read().strip()

    rows = load_jsonl(args.dataset)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if args.resume and out_path.exists():
        for line in open(out_path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                done.add(sample_key(json.loads(line)))
            except Exception:
                pass
    print(f"[RESUME] existing_done={len(done)}", flush=True)

    write_lock = threading.Lock()
    stats_lock = threading.Lock()
    stats = {"done": len(done), "valid": 0, "failed": 0, "slow": 0}
    start = time.time()

    mode = "a" if (args.resume and out_path.exists()) else "w"
    wf = open(out_path, mode, encoding="utf-8")

    def process(item):
        key = sample_key(item)
        if key in done:
            return
        user_prompt = build_question_text(item, trailing)
        try:
            raw = call_api(candidate, system_prompt, user_prompt,
                           args.timeout, args.max_tokens, 1.5, 2.0, 60.0, 0.3)
            pred = extract_answer_letter(raw)
            if pred in OPTION_LETTERS:
                out = dict(item)
                out["TeacherAnswer"] = pred
                out["TeacherRaw"] = raw
                out["OriginalAnswer"] = str(item.get("Answer") or item.get("answer") or "").strip().upper()
                out["Answer"] = pred
                line = json.dumps(out, ensure_ascii=False) + "\n"
                with write_lock:
                    wf.write(line)
                    wf.flush()
                with stats_lock:
                    stats["valid"] += 1
            else:
                with stats_lock:
                    stats["failed"] += 1
        except Exception as e:
            with stats_lock:
                stats["failed"] += 1
            err = {"error": str(e)[:200],
                   "question": str(item.get("Question") or "")[:100]}
            print("[ERR]", json.dumps(err, ensure_ascii=False), flush=True)
        finally:
            with stats_lock:
                stats["done"] += 1
                if stats["done"] % 100 == 0:
                    el = time.time() - start
                    rate = stats["done"] / max(el, 1)
                    eta = (len(rows) - stats["done"]) / max(rate, 1e-9) / 3600
                    print(f"[PROGRESS] {stats['done']}/{len(rows)} "
                          f"valid={stats['valid']} failed={stats['failed']} "
                          f"rate={rate:.2f}/s eta={eta:.1f}h", flush=True)

    if args.workers <= 1:
        for item in rows:
            process(item)
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(process, rows))

    wf.close()
    print(f"[DONE] total={stats['done']} valid={stats['valid']} "
          f"failed={stats['failed']} output={out_path}", flush=True)
    if stats["valid"] == 0 and len(done) == 0:
        print("[FATAL] no valid labels", flush=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
