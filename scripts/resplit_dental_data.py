#!/usr/bin/env python3
"""
Resplit cmexam_full.csv dental subset into new train/val/test sets.
Uses stratified split by difficulty level for balanced distribution.
Target ratio: 70/15/15 → larger test set for lower variance.

Usage:
    python scripts/resplit_dental_data.py
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
OUTPUT_DIR = ROOT / "shared" / "splits_v2"


def load_dental_questions(csv_path):
    """Load single-answer dental questions from cmexam_full.csv."""
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    dental = [
        r for r in rows
        if r.get("Medical Discipline", "") == "口腔医学"
        and len(r.get("Answer", "")) == 1
    ]
    print(f"Loaded {len(dental)} dental single-answer questions from {len(rows)} total")
    return dental


def stratified_split(data, seed=SEED):
    """Stratified split by difficulty level."""
    random.seed(seed)

    # Group by difficulty
    by_difficulty = defaultdict(list)
    for row in data:
        diff = row.get("Difficulty level", "unknown")
        by_difficulty[diff].append(row)

    train, val, test = [], [], []

    for diff, rows in sorted(by_difficulty.items()):
        random.shuffle(rows)
        n = len(rows)
        n_val = max(1, round(n * VAL_RATIO))
        n_test = max(1, round(n * TEST_RATIO))
        n_train = n - n_val - n_test

        train.extend(rows[:n_train])
        val.extend(rows[n_train:n_train + n_val])
        test.extend(rows[n_train + n_val:])

    # Shuffle within each split
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    return train, val, test


def save_jsonl(data, path):
    """Save list of dicts as JSONL."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  Saved {len(data)} rows → {path}")


def print_stats(name, data):
    """Print distribution statistics."""
    diff_dist = Counter(r.get("Difficulty level", "?") for r in data)
    dept_dist = Counter(r.get("Clinical Department", "?") for r in data)
    ans_dist = Counter(r.get("Answer", "?") for r in data)
    print(f"\n  [{name}] n={len(data)}")
    print(f"    Difficulty: {dict(sorted(diff_dist.items()))}")
    print(f"    Answer dist: {dict(sorted(ans_dist.items()))}")
    print(f"    Top depts: {dept_dist.most_common(5)}")


def main():
    dental = load_dental_questions(CSV_PATH)
    train, val, test = stratified_split(dental)

    print(f"\n=== New Split (seed={SEED}) ===")
    print(f"Train: {len(train)} ({len(train)/len(dental)*100:.1f}%)")
    print(f"Val:   {len(val)} ({len(val)/len(dental)*100:.1f}%)")
    print(f"Test:  {len(test)} ({len(test)/len(dental)*100:.1f}%)")
    print(f"Total: {len(train)+len(val)+len(test)}")

    print_stats("Train", train)
    print_stats("Val", val)
    print_stats("Test", test)

    # Check overlap
    train_qs = set(r["Question"] for r in train)
    val_qs = set(r["Question"] for r in val)
    test_qs = set(r["Question"] for r in test)
    assert len(train_qs & val_qs) == 0, "Train/Val overlap!"
    assert len(train_qs & test_qs) == 0, "Train/Test overlap!"
    assert len(val_qs & test_qs) == 0, "Val/Test overlap!"
    print("\n✓ No overlap between splits")

    # Save
    save_jsonl(train, OUTPUT_DIR / "train.jsonl")
    save_jsonl(val, OUTPUT_DIR / "val.jsonl")
    save_jsonl(test, OUTPUT_DIR / "test.jsonl")

    # Check teacher label coverage
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
    train_missing = len(train) - train_covered
    print(f"\nTeacher label coverage for new train set:")
    print(f"  Covered: {train_covered}/{len(train)}")
    print(f"  Missing: {train_missing} (need API calls)")

    # Save list of questions needing teacher labels
    missing = [r for r in train if r["Question"] not in teacher_qs]
    if missing:
        save_jsonl(missing, OUTPUT_DIR / "train_need_teacher.jsonl")


if __name__ == "__main__":
    main()
