#!/usr/bin/env python3
"""调用教师模型 API 为每道题生成 Chain-of-Thought 推理链。

输出 JSONL 每行包含原始字段 + Rationale + RationaleAnswer。
仅保留 RationaleAnswer 能成功提取且与 GT 匹配的样本。

用法:
    python generate_teacher_cot.py \
        --dataset data/train.jsonl \
        --candidate configs/teacher_candidate.json \
        --output data/train_cot.jsonl \
        --gt_field Answer \
        --resume
"""
import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import requests

ANSWER_RE = re.compile(r"答案[：:]\s*([A-EＡＢＣＤＥ])", re.IGNORECASE)
FALLBACK_RE = re.compile(r"\b([A-E])\b")

COT_SYSTEM_PROMPT = (
    "你是一位专业的牙科医生。请对以下牙科单项选择题进行逐步分析推理。\n"
    "要求：\n"
    "1. 先分析题目涉及的知识点\n"
    "2. 逐一分析各选项的正确性\n"
    "3. 给出推理结论\n"
    "4. 最后一行必须以\"答案：X\"结尾（X为A/B/C/D/E中的一个字母）"
)


def resolve_api_key(candidate):
    direct_key = str(candidate.get("api_key", "")).strip()
    if direct_key:
        return direct_key
    key_env = str(candidate.get("api_key_env", "")).strip()
    if not key_env:
        return ""
    if key_env.startswith("sk-") or key_env.startswith("eyJ"):
        return key_env
    if re.fullmatch(r"[A-Z_][A-Z0-9_]*", key_env):
        return os.getenv(key_env, "").strip()
    return key_env


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_question_text(item):
    q = str(item.get("Question") or item.get("question") or "").strip()
    options = item.get("Options") or item.get("options") or {}
    lines = [q]
    if isinstance(options, dict):
        for k in ["A", "B", "C", "D", "E"]:
            if k in options:
                lines.append(f"{k}. {str(options[k]).strip()}")
    else:
        opt_text = str(options).strip()
        if opt_text:
            lines.append(opt_text)
    return "\n".join(lines)


def sample_key(item):
    q = str(item.get("Question") or item.get("question") or "").strip()
    options = item.get("Options") or item.get("options") or {}
    if isinstance(options, dict):
        opt_text = "\n".join(f"{k}:{str(options.get(k, ''))}" for k in ["A", "B", "C", "D", "E"])
    else:
        opt_text = str(options).strip()
    raw = f"{q}\n{opt_text}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def extract_cot_answer(text):
    """从 CoT 输出中提取答案字母。优先匹配 '答案：X' 格式。"""
    if not text:
        return ""
    m = ANSWER_RE.search(text)
    if m:
        ch = m.group(1).strip().upper()
        # Handle fullwidth letters
        fw_map = {"Ａ": "A", "Ｂ": "B", "Ｃ": "C", "Ｄ": "D", "Ｅ": "E"}
        return fw_map.get(ch, ch)
    # Fallback: last occurrence of A-E in text
    matches = FALLBACK_RE.findall(text.upper())
    return matches[-1] if matches else ""


def resolve_base_url(candidate):
    base_url = str(candidate.get("base_url") or candidate.get("api_base") or "").strip()
    if not base_url:
        raise RuntimeError("missing base_url/api_base in candidate config")
    if "chat/completions" not in base_url:
        base_url = base_url.rstrip("/") + "/chat/completions"
    return base_url


def call_api(candidate, system_prompt, user_prompt, timeout_sec=180,
             max_tokens=1024, max_retries=4):
    api_key = resolve_api_key(candidate)
    if not api_key:
        raise RuntimeError("missing api key")
    base_url = resolve_base_url(candidate)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": candidate["model"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
    }

    last_err = None
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(base_url, headers=headers, json=payload, timeout=timeout_sec)
            if resp.status_code >= 400:
                text = (resp.text or "").strip().replace("\n", " ")
                raise RuntimeError(f"http_{resp.status_code}: {text[:240]}")
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                err_text = str(e).lower()
                if "429" in err_text or "rate limit" in err_text:
                    time.sleep(60 + random.uniform(0, 5))
                else:
                    time.sleep(2.0 * (2 ** attempt) + random.uniform(0, 1))
                continue
            raise last_err
    raise last_err


def main():
    parser = argparse.ArgumentParser(description="生成教师 CoT 推理链")
    parser.add_argument("--dataset", required=True, help="输入 JSONL 数据集")
    parser.add_argument("--candidate", required=True, help="教师模型配置 JSON")
    parser.add_argument("--output", required=True, help="输出 JSONL")
    parser.add_argument("--gt_field", default="Answer",
                        help="GT 答案字段名 (default: Answer)")
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--timeout_sec", type=int, default=180)
    parser.add_argument("--max_retries", type=int, default=4)
    parser.add_argument("--request_interval_sec", type=float, default=0.5)
    parser.add_argument("--filter_correct", action="store_true",
                        help="仅保留 CoT 答案与 GT 一致的样本")
    parser.add_argument("--resume", action="store_true",
                        help="断点续传：跳过已处理的样本")
    args = parser.parse_args()

    with open(args.candidate, "r", encoding="utf-8") as f:
        candidate = json.load(f)

    rows = load_jsonl(args.dataset)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_keys = set()
    if args.resume and out_path.exists():
        for line in open(out_path, "r", encoding="utf-8"):
            line = line.strip()
            if line:
                try:
                    done_keys.add(sample_key(json.loads(line)))
                except Exception:
                    pass
        print(f"[RESUME] 已有 {len(done_keys)} 条记录", flush=True)

    total = 0
    valid = 0
    match_gt = 0
    failed = 0

    mode = "a" if (args.resume and out_path.exists()) else "w"
    with open(out_path, mode, encoding="utf-8") as wf:
        for i, item in enumerate(rows, start=1):
            if sample_key(item) in done_keys:
                continue

            gt = str(item.get(args.gt_field, "")).strip().upper()
            prompt = build_question_text(item)
            total += 1

            try:
                raw = call_api(
                    candidate=candidate,
                    system_prompt=COT_SYSTEM_PROMPT,
                    user_prompt=prompt,
                    timeout_sec=args.timeout_sec,
                    max_tokens=args.max_tokens,
                    max_retries=args.max_retries,
                )
                pred = extract_cot_answer(raw)

                if pred in {"A", "B", "C", "D", "E"}:
                    valid += 1
                    gt_match = (pred == gt)
                    if gt_match:
                        match_gt += 1

                    if args.filter_correct and not gt_match:
                        pass  # skip non-matching
                    else:
                        out_item = dict(item)
                        out_item["Rationale"] = raw
                        out_item["RationaleAnswer"] = pred
                        out_item["GTMatch"] = gt_match
                        wf.write(json.dumps(out_item, ensure_ascii=False) + "\n")
                        wf.flush()
                        done_keys.add(sample_key(item))
                else:
                    failed += 1
                    print(f"[WARN] #{i}: 无法提取答案: {raw[:80]}...", flush=True)

            except Exception as e:
                failed += 1
                print(f"[ERR] #{i}: {e}", flush=True)

            if args.request_interval_sec > 0:
                time.sleep(args.request_interval_sec)

            if i % 20 == 0:
                print(
                    f"[PROGRESS] {i}/{len(rows)} valid={valid} match_gt={match_gt} "
                    f"failed={failed} acc={match_gt/valid*100:.1f}%" if valid > 0 else
                    f"[PROGRESS] {i}/{len(rows)} valid={valid} failed={failed}",
                    flush=True,
                )

    total_done = len(done_keys)
    print(f"\n[DONE] total_processed={total} valid={valid} match_gt={match_gt} "
          f"failed={failed} output_total={total_done}")
    if valid > 0:
        print(f"[DONE] CoT accuracy={match_gt/valid*100:.1f}%")


if __name__ == "__main__":
    main()
