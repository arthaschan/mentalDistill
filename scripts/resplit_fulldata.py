#!/usr/bin/env python3
"""
Resplit ALL single-answer CMExam questions into train/val/test (70/15/15).
Stratified by Medical Discipline + Difficulty level for balanced distribution.
Also tags dental subset for dual-evaluation.

Usage:
    python scripts/resplit_fulldata.py
"""
import csv
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

SEED = 2026
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "shared" / "cmexam_full.csv"
OUTPUT_DIR = ROOT / "15_fulldata_resplit" / "data"


def load_single_answer(csv_path):
    """Load all single-answer questions, deduplicated by Question text."""
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    single = [r for r in rows if len(r.get("Answer", "")) == 1]
    # Deduplicate: keep first occurrence
    seen = set()
    deduped = []
    for r in single:
        q = r["Question"]
        if q not in seen:
            seen.add(q)
            deduped.append(r)
    print(f"Loaded {len(deduped)} unique single-answer from {len(rows)} total ({len(single)-len(deduped)} dups removed)")
    return deduped


def stratified_split(data, seed=SEED):
    """Stratified split by (Medical Discipline, Difficulty level)."""
    random.seed(seed)

    # Group by composite key
    by_stratum = defaultdict(list)
    for row in data:
        disc = row.get("Medical Discipline", "other")
        diff = row.get("Difficulty level", "0")
        by_stratum[(disc, diff)].append(row)

    train, val, test = [], [], []

    for key, rows in sorted(by_stratum.items()):
        random.shuffle(rows)
        n = len(rows)
        n_val = max(1, round(n * VAL_RATIO))
        n_test = max(1, round(n * TEST_RATIO))
        n_train = n - n_val - n_test
        if n_train < 0:
            # Very small stratum: put all in train
            train.extend(rows)
            continue

        train.extend(rows[:n_train])
        val.extend(rows[n_train:n_train + n_val])
        test.extend(rows[n_train + n_val:])

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    return train, val, test


def save_jsonl(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  Saved {len(data)} rows → {path}")


def extract_dental_subset(data):
    """Extract dental (口腔医学) subset."""
    return [r for r in data if r.get("Medical Discipline", "") == "口腔医学"]


def print_stats(name, data):
    disc_dist = Counter(r.get("Medical Discipline", "?") for r in data)
    diff_dist = Counter(r.get("Difficulty level", "?") for r in data)
    dental_n = sum(1 for r in data if r.get("Medical Discipline", "") == "口腔医学")
    print(f"\n  [{name}] n={len(data)}, dental={dental_n}")
    print(f"    Discipline: {dict(sorted(disc_dist.items()))}")
    print(f"    Difficulty: {dict(sorted(diff_dist.items()))}")


def main():
    data = load_single_answer(CSV_PATH)
    train, val, test = stratified_split(data)

    total = len(train) + len(val) + len(test)
    print(f"\n=== Full-Data Split (seed={SEED}) ===")
    print(f"Train: {len(train)} ({len(train)/total*100:.1f}%)")
    print(f"Val:   {len(val)} ({len(val)/total*100:.1f}%)")
    print(f"Test:  {len(test)} ({len(test)/total*100:.1f}%)")
    print(f"Total: {total}")

    print_stats("Train", train)
    print_stats("Val", val)
    print_stats("Test", test)

    # No overlap check
    train_qs = set(r["Question"] for r in train)
    val_qs = set(r["Question"] for r in val)
    test_qs = set(r["Question"] for r in test)
    assert len(train_qs & val_qs) == 0, "Train/Val overlap!"
    assert len(train_qs & test_qs) == 0, "Train/Test overlap!"
    assert len(val_qs & test_qs) == 0, "Val/Test overlap!"
    print("\n✓ No overlap between splits")

    # Save full splits
    save_jsonl(train, OUTPUT_DIR / "train.jsonl")
    save_jsonl(val, OUTPUT_DIR / "val.jsonl")
    save_jsonl(test, OUTPUT_DIR / "test.jsonl")

    # Save dental subsets for dual evaluation
    dental_train = extract_dental_subset(train)
    dental_val = extract_dental_subset(val)
    dental_test = extract_dental_subset(test)
    save_jsonl(dental_test, OUTPUT_DIR / "test_dental.jsonl")
    save_jsonl(dental_val, OUTPUT_DIR / "val_dental.jsonl")
    print(f"\n  Dental test subset: {len(dental_test)} questions")
    print(f"  Dental val subset:  {len(dental_val)} questions")
    print(f"  Dental train subset: {len(dental_train)} questions")

    # Check existing teacher label coverage
    teacher_qs = set()
    for tf in [
        ROOT / "02_deepseek_v3_choice_head" / "data" / "teacher_train.jsonl",
        ROOT / "02_deepseek_v3_choice_head" / "data" / "teacher_test.jsonl",
    ]:
        if tf.exists():
            with open(tf) as fh:
                for line in fh:
                    r = json.loads(line)
                    teacher_qs.add(r["Question"])

    train_covered = sum(1 for r in train if r["Question"] in teacher_qs)
    print(f"\nExisting DeepSeek teacher coverage for new train:")
    print(f"  Covered: {train_covered}/{len(train)}")
    print(f"  Need new labels: {len(train) - train_covered}")

    # Save the questions needing teacher labels
    need_labels = [r for r in train if r["Question"] not in teacher_qs]
    if need_labels:
        save_jsonl(need_labels, OUTPUT_DIR / "train_need_teacher.jsonl")


if __name__ == "__main__":
    main()
