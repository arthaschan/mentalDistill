#!/usr/bin/env python3
"""
数据来源标注 & 质量审计脚本

功能:
1. 对 mentalDistill 各模块 data/ 下的 jsonl 做来源标注 (source 字段)
2. 基于 EasyEdit3/data/ 中的原始数据集做匹配:
   - cmexam: CMExam 原始试题
   - deepseek_autogen: DeepSeek 自动生成
   - huatuo_converted: 华佗数据集转换
   - unknown: 无法匹配到已知来源
3. 输出质量审计报告 (选项重复、选项数异常、极短题干等)
4. 可选: 修复已知硬伤样本

用法:
  python3 shared/audit_and_tag_source.py --easyedit-root ../EasyEdit3 [--fix] [--dry-run]
"""

import argparse
import json
import os
import sys
from pathlib import Path


def load_jsonl(path):
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


def build_source_index(easyedit_root):
    """Build question -> source mapping from EasyEdit3 data."""
    index = {}
    root = Path(easyedit_root) / "data"

    # CMExam originals
    for split in ["train", "val", "test"]:
        path = root / f"cmexam_dental_choice_{split}.jsonl"
        if path.exists():
            for d in load_jsonl(path):
                index.setdefault(d["Question"], []).append("cmexam")

    # DeepSeek autogen
    for fname in ["augment/deepseek_autogen_mcq.jsonl",
                   "augment/autogen_train.jsonl",
                   "augment/autogen_val.jsonl",
                   "augment/autogen_test.jsonl"]:
        path = root / fname
        if path.exists():
            for d in load_jsonl(path):
                index.setdefault(d["Question"], []).append("deepseek_autogen")

    # Huatuo converted
    for fname in ["augment/huatuo_dental_mcq.jsonl",
                   "augment/huatuo_train.jsonl",
                   "augment/huatuo_val.jsonl",
                   "augment/huatuo_test.jsonl"]:
        path = root / fname
        if path.exists():
            for d in load_jsonl(path):
                index.setdefault(d["Question"], []).append("huatuo_converted")

    # Deduplicate sources per question
    for q in index:
        index[q] = sorted(set(index[q]))

    return index


def audit_sample(d, line_num, filepath):
    """Run quality checks on a single sample. Returns list of issues."""
    issues = []
    q = d.get("Question", "")
    opts = d.get("Options", "")
    ans = d.get("Answer", "").strip()
    exp = d.get("Explanation", "")

    # Answer validity
    if ans not in ["A", "B", "C", "D", "E"]:
        issues.append(f"{filepath}:{line_num} invalid answer: [{ans}]")

    # Option parsing
    opt_lines = [o.strip() for o in opts.split("\n") if o.strip()]
    opt_letters = [o[0] for o in opt_lines if o]
    opt_texts = [o[2:].strip() for o in opt_lines if len(o) > 2]

    # Answer not in options
    if ans and ans not in opt_letters:
        issues.append(f"{filepath}:{line_num} answer {ans} not in options")

    # Option count
    if len(opt_lines) < 5:
        issues.append(f"{filepath}:{line_num} only {len(opt_lines)} options (expected 5)")

    # Duplicate option texts
    if len(opt_texts) != len(set(opt_texts)):
        issues.append(f"{filepath}:{line_num} duplicate option texts")

    # Very short explanation
    if len(exp) < 10:
        issues.append(f"{filepath}:{line_num} very short explanation ({len(exp)} chars)")

    # Very short question
    if len(q) < 8:
        issues.append(f"{filepath}:{line_num} very short question ({len(q)} chars): {q}")

    # Difficulty range
    dl = d.get("Difficulty level")
    if dl is not None and dl not in [1, 2, 3, 4, 5]:
        issues.append(f"{filepath}:{line_num} invalid difficulty: {dl}")

    return issues


KNOWN_FIXES = {
    # train line 92: D and E are both "异丙嗪", fix E to "西咪替丁"
    "十二指肠溃疡的首选为": {
        "Options": "A 雷尼替丁\nB 氨茶碱\nC 苯海拉明\nD 异丙嗪\nE 西咪替丁"
    },
}


def process_file(filepath, source_index, apply_fix=False, dry_run=False):
    """Tag source and audit a single jsonl file. Returns (tagged_data, issues, stats)."""
    data = load_jsonl(filepath)
    issues = []
    stats = {"cmexam": 0, "deepseek_autogen": 0, "huatuo_converted": 0, "unknown": 0, "fixed": 0}

    for i, d in enumerate(data):
        line_num = i + 1
        q = d["Question"]

        # Tag source
        sources = source_index.get(q, ["unknown"])
        d["source"] = sources[0]  # primary source
        stats[sources[0]] = stats.get(sources[0], 0) + 1

        # Audit
        sample_issues = audit_sample(d, line_num, filepath)
        issues.extend(sample_issues)

        # Apply known fixes
        if apply_fix and q in KNOWN_FIXES:
            for k, v in KNOWN_FIXES[q].items():
                d[k] = v
            stats["fixed"] += 1

    if not dry_run:
        save_jsonl(data, filepath)

    return data, issues, stats


def main():
    parser = argparse.ArgumentParser(description="数据来源标注与质量审计")
    parser.add_argument("--easyedit-root", required=True, help="EasyEdit3 项目根目录")
    parser.add_argument("--fix", action="store_true", help="应用已知硬伤修复")
    parser.add_argument("--dry-run", action="store_true", help="仅报告，不修改文件")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    easyedit_root = Path(args.easyedit_root).resolve()

    if not (easyedit_root / "data").exists():
        print(f"Error: {easyedit_root}/data/ not found")
        sys.exit(1)

    print(f"Building source index from {easyedit_root}/data/ ...")
    source_index = build_source_index(easyedit_root)
    print(f"  Indexed {len(source_index)} unique questions")

    # Find all module data dirs
    modules = sorted(project_root.glob("[0-9][0-9]_*/data"))
    all_issues = []
    total_stats = {}

    for mod_data in modules:
        mod_name = mod_data.parent.name
        for split in ["train.jsonl", "val.jsonl", "test.jsonl"]:
            fpath = mod_data / split
            if not fpath.exists():
                continue
            _, issues, stats = process_file(
                str(fpath), source_index,
                apply_fix=args.fix, dry_run=args.dry_run
            )
            all_issues.extend(issues)
            for k, v in stats.items():
                total_stats[k] = total_stats.get(k, 0) + v
            src_summary = {k: v for k, v in stats.items() if k != "fixed" and v > 0}
            action = "DRY-RUN" if args.dry_run else "TAGGED"
            print(f"  [{action}] {mod_name}/{split}: {src_summary}")

    # Summary
    print(f"\n=== Quality Issues ({len(all_issues)}) ===")
    for iss in all_issues:
        print(f"  {iss}")

    print(f"\n=== Source Distribution ===")
    for k, v in sorted(total_stats.items()):
        if k != "fixed":
            print(f"  {k}: {v}")
    if args.fix:
        print(f"  fixed: {total_stats.get('fixed', 0)}")

    if args.dry_run:
        print("\n(dry-run mode: no files were modified)")


if __name__ == "__main__":
    main()
