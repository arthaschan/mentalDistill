#!/usr/bin/env python3
"""
precheck_option_tokens.py — 阶段 2 教师 token 映射预检

阶段 2 最常见的坑：不同 tokenizer 对 "A"/" A"/"\\nA" 的编码不同。若 A/B/C/D/E
不能各自映射到一个【唯一且解码回自身】的 token，那么从 logits 提取的选项概率
就是错的（可能取到子词、合并 token、或与别的选项撞 id）。本脚本在【正式生成
logprobs 前】对每个新教师做严格校验，只下载/加载 tokenizer（不加载权重，纯 CPU）。

校验项（每个字母 A-E）：
  1. 能否找到映射 token id（复用 generate_teacher_labels_local_logprobs.get_option_token_ids 逻辑）
  2. 该 id 解码回的字符串 strip 后是否 == 字母本身（防子词/脏 token）
  3. 5 个字母的 token id 是否两两不同（防撞 id）
  4. 额外探针：在真实 chat 模板下喂一道样题，看模型 assistant 起始位置的 next-token
     argmax 是否落在 5 个选项 token 之一（确认"答案位置"对齐）——可选，需要权重时跳过

退出码：全部教师通过=0；任一不通过=1（便于脚本串联时阻断）。
"""
import argparse
import json
import sys
from pathlib import Path

OPTION_LETTERS = ["A", "B", "C", "D", "E"]
# file is at research/distillability/scripts/, so repo root is parents[3]
REPO_ROOT = Path(__file__).resolve().parents[3]


def get_option_token_ids(tokenizer):
    """与 shared/generate_teacher_labels_local_logprobs.py 完全一致的映射逻辑。"""
    ids = {}
    for letter in OPTION_LETTERS:
        direct = tokenizer.encode(letter, add_special_tokens=False)
        if len(direct) == 1:
            ids[letter] = direct[0]
        else:
            space_encoded = tokenizer.encode(f" {letter}", add_special_tokens=False)
            ids[letter] = space_encoded[-1]
    return ids


def encoding_report(tokenizer, letter):
    """返回三种编码方式的 token 序列，用于诊断。"""
    out = {}
    for tag, text in [("bare", letter), ("space", f" {letter}"), ("newline", f"\n{letter}")]:
        out[tag] = tokenizer.encode(text, add_special_tokens=False)
    return out


def check_tokenizer(label, model_dir):
    from transformers import AutoTokenizer
    result = {"label": label, "model_dir": str(model_dir), "passed": False, "issues": [], "mapping": {}}
    try:
        tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    except Exception as e:
        result["issues"].append(f"tokenizer load failed: {e}")
        return result

    ids = get_option_token_ids(tok)
    result["mapping"] = {k: int(v) for k, v in ids.items()}

    # check 2: decode round-trip
    for letter in OPTION_LETTERS:
        decoded = tok.decode([ids[letter]]).strip()
        if decoded != letter:
            enc = encoding_report(tok, letter)
            result["issues"].append(
                f"{letter}: token id {ids[letter]} decodes to '{decoded}' (!= '{letter}'); encodings={enc}")

    # check 3: distinct ids
    id_list = [ids[l] for l in OPTION_LETTERS]
    if len(set(id_list)) != len(id_list):
        dupes = {l: ids[l] for l in OPTION_LETTERS}
        result["issues"].append(f"option token ids not all distinct: {dupes}")

    # extra info: multi-token letters (warning, not hard fail if round-trip ok)
    multi = {}
    for letter in OPTION_LETTERS:
        direct = tok.encode(letter, add_special_tokens=False)
        if len(direct) != 1:
            multi[letter] = direct
    if multi:
        result["warnings"] = result.get("warnings", [])
        result["warnings"].append(f"letters not single-token when bare-encoded (fell back to ' X'): {multi}")

    result["passed"] = len(result["issues"]) == 0
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teachers", nargs="+",
                    help="label:model_dir pairs. If omitted, uses the 4 Phase-2 defaults.")
    ap.add_argument("--output", default="research/distillability/outputs/option_token_precheck.json")
    args = ap.parse_args()

    if args.teachers:
        specs = [s.split(":", 1) for s in args.teachers]
    else:
        m = REPO_ROOT / "models"
        specs = [
            ("phi-4", str(m / "phi-4")),
            ("gemma-2-27b-it", str(m / "gemma-2-27b-it")),
            ("GLM-4-32B-0414", str(m / "GLM-4-32B-0414")),
            ("Yi-1.5-34B-Chat", str(m / "Yi-1.5-34B-Chat")),
        ]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    reports = []
    print("=" * 84)
    print("Phase-2 option-token mapping precheck (A/B/C/D/E)")
    print("=" * 84)
    any_fail = False
    for label, mdir in specs:
        if not Path(mdir, "config.json").exists() and not Path(mdir, "tokenizer_config.json").exists():
            print(f"\n--- {label}: NOT DOWNLOADED YET ({mdir}) — skipped")
            reports.append({"label": label, "model_dir": mdir, "passed": None, "issues": ["not downloaded"]})
            continue
        r = check_tokenizer(label, mdir)
        reports.append(r)
        status = "PASS" if r["passed"] else "FAIL"
        if not r["passed"]:
            any_fail = True
        print(f"\n--- {label}: {status}")
        print(f"  mapping: {r['mapping']}")
        for w in r.get("warnings", []):
            print(f"  [warn] {w}")
        for iss in r["issues"]:
            print(f"  [issue] {iss}")

    json.dump({"reports": reports}, open(args.output, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n[SAVED] {args.output}")

    ready = [r for r in reports if r["passed"] is not None]
    passed = [r for r in ready if r["passed"]]
    print(f"\nSummary: {len(passed)}/{len(ready)} downloaded teachers passed; "
          f"{len([r for r in reports if r['passed'] is None])} not yet downloaded.")
    if any_fail:
        print("ACTION: failed teachers need a custom option-token strategy before logprob generation.")
        sys.exit(1)


if __name__ == "__main__":
    main()
