#!/usr/bin/env python3
"""对抗式辩论：在共识错题（两老师都错）上，注入"怀疑"看能否救回正确。

做法（2 智能体，各自独立接到同一对抗提示）：
  - 告诉每个教授"之前两位专家答了 [ans1]/[ans2]，但可能共享盲点、都可能错"，
    要求重新从零解、并怀疑前答。
  - rescued = 任一教授新答案 == GT。

用法：
  python adversarial_rescue.py --limit 50 --workers 6 --resume
  输出 data/rescue_results.jsonl + 打印救回率。
"""
import argparse
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

import requests

OPT = "ABCDE"
ANSWER_RE = re.compile(r"ANSWER\s*[:：]\s*([A-E])", re.I)
LAST_RE = re.compile(r"\b([A-E])\b")

DSV = {
    "name": "dsv4pro", "model": "deepseek-v4-pro",
    "base": "https://api.deepseek.com/v1", "key_env": "DEEPSEEK_API_KEY",
}
DOB = {
    "name": "doubao", "model": "doubao-seed-2-1-turbo-260628",
    "base": "https://ark.cn-beijing.volces.com/api/v3", "key_env": "DOUBAO_API_KEY",
}


def key_of(c):
    env = c["key_env"]
    return os.environ.get(env, "")


def base_of(c):
    b = c["base"].rstrip("/")
    return b + ("/chat/completions" if "chat/completions" not in b else "")


def build_prompt(q, opts, ans1, ans2):
    if isinstance(opts, dict):
        opt_text = "\n".join(f"{k}. {opts[k]}" for k in OPT if k in opts)
    else:
        opt_text = str(opts)
    hint = (f"Two previous experts answered {ans1} and {ans2}. A skeptical reviewer "
            f"suspects they may share a common blind spot and could BOTH be wrong.")
    return (f"{q}\n{opt_text}\n\n{hint}\n"
            f"Re-solve this question from scratch. Be skeptical of {ans1} and {ans2}. "
            f"Give your step-by-step reasoning, then end with 'ANSWER: X' "
            f"where X is A/B/C/D/E.")


def call(c, prompt, timeout, max_tokens):
    key = key_of(c)
    if not key:
        raise RuntimeError(f"missing key {c['key_env']}")
    sys_msg = ("You are a medical expert serving as a skeptical second-opinion "
               "reviewer.")
    payload = {"model": c["model"],
               "messages": [{"role": "system", "content": sys_msg},
                            {"role": "user", "content": prompt}],
               "temperature": 0, "max_tokens": max_tokens}
    for attempt in range(4):
        try:
            r = requests.post(base_of(c),
                              headers={"Authorization": f"Bearer {key}",
                                       "Content-Type": "application/json"},
                              json=payload, timeout=timeout)
            if r.status_code >= 400:
                raise RuntimeError(f"http_{r.status_code}: {r.text[:160]}")
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt == 3:
                raise
            time.sleep(2.0 * (2 ** attempt))


def extract(text):
    if not text:
        return ""
    m = ANSWER_RE.search(text)
    if m:
        return m.group(1).upper()
    ms = LAST_RE.findall(text.upper())
    return ms[-1] if ms else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--agents", default="dsv4pro,doubao",
                    help="逗号分隔: dsv4pro,doubao（默认）或只 dsv4pro")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    agents = {"dsv4pro": DSV, "doubao": DOB}
    use_agents = [agents[a] for a in args.agents.split(",") if a in agents]

    rows = [json.loads(l) for l in open(
        "26_collaborative_distill/data/consensus_errors.jsonl") if l.strip()]
    if args.limit:
        rows = rows[:args.limit]

    out_path = Path("26_collaborative_distill/data/rescue_results.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.resume and out_path.exists():
        for l in open(out_path):
            l = l.strip()
            if l:
                try:
                    done.add(json.loads(l)["uid"])
                except Exception:
                    pass

    write_lock = threading.Lock()
    stats = {"n": 0, "rescued": 0, "fail": 0}
    start = time.time()

    def process(row):
        if row["uid"] in done:
            return
        prompt = build_prompt(row["Question"], row["Options"],
                              row["dsv4pro"], row["doubao"])
        gt = row["GT"]
        out = {"uid": row["uid"], "GT": gt,
               "prev_dsv4pro": row["dsv4pro"], "prev_doubao": row["doubao"],
               "same_wrong": row["same_wrong"]}
        try:
            for c in use_agents:
                raw = call(c, prompt, args.timeout, args.max_tokens)
                out[c["name"]] = extract(raw)
                out[c["name"] + "_raw"] = raw[:400]
        except Exception as e:
            out["error"] = str(e)[:200]
        out["rescued"] = (out.get("dsv4pro") == gt or out.get("doubao") == gt)
        line = json.dumps(out, ensure_ascii=False) + "\n"
        with write_lock:
            with open(out_path, "a") as f:
                f.write(line)
                f.flush()
            stats["n"] += 1
            if out["rescued"]:
                stats["rescued"] += 1
            if "error" in out:
                stats["fail"] += 1
            if stats["n"] % 10 == 0:
                el = time.time() - start
                rate = stats["n"] / max(el, 1)
                print(f"[PROGRESS] {stats['n']}/{len(rows)} "
                      f"rescued={stats['rescued']} "
                      f"fail={stats['fail']} rate={rate:.2f}/s", flush=True)

    if args.workers <= 1:
        for r in rows:
            process(r)
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(process, rows))

    # 汇总
    done_rows = [json.loads(l) for l in open(out_path) if l.strip()]
    n = len(done_rows)
    rescued = sum(1 for r in done_rows if r.get("rescued"))
    print(f"\n=== 对抗式辩论救回率 ===", flush=True)
    print(f"  共识错题 {n} 道, 救回 {rescued} 道 = {100.0*rescued/n:.1f}%", flush=True)
    same = [r for r in done_rows if r.get("same_wrong")]
    same_resc = sum(1 for r in same if r.get("rescued"))
    print(f"  其中'错同一个答案' {len(same)} 道, 救回 {same_resc} "
          f"= {100.0*same_resc/len(same):.1f}%" if same else "", flush=True)


if __name__ == "__main__":
    main()
