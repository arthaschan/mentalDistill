#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


VALID_LABELS = {"A", "B", "C", "D", "E"}


def normalize_label(value):
    text = str(value or "").strip().upper()
    for char in text:
        if char in VALID_LABELS:
            return char
    return ""


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize(rows):
    total = 0
    parsed = 0
    correct = 0

    for item in rows:
        ground_truth = normalize_label(item.get("OriginalAnswer"))
        if not ground_truth:
            ground_truth = normalize_label(item.get("Answer"))

        prediction = normalize_label(item.get("TeacherAnswer"))
        if not prediction and item.get("OriginalAnswer"):
            prediction = normalize_label(item.get("Answer"))

        if not ground_truth:
            continue

        total += 1
        if prediction:
            parsed += 1
            if prediction == ground_truth:
                correct += 1

    accuracy = (correct / total * 100.0) if total else 0.0
    parsed_ratio = (parsed / total * 100.0) if total else 0.0
    mismatch = total - correct
    mismatch_ratio = (mismatch / total * 100.0) if total else 0.0

    return {
        "total": total,
        "parsed": parsed,
        "parsed_ratio": round(parsed_ratio, 2),
        "correct": correct,
        "accuracy": round(accuracy, 2),
        "mismatch": mismatch,
        "mismatch_ratio": round(mismatch_ratio, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Summarize teacher label accuracy from a JSONL file.")
    parser.add_argument("--input", required=True, help="Teacher label JSONL path")
    parser.add_argument("--report", default=None, help="Optional JSON report output path")
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    report = summarize(rows)

    print("Teacher Eval Summary")
    print(f"- input: {args.input}")
    print(f"- total: {report['total']}")
    print(f"- parsed: {report['parsed']} ({report['parsed_ratio']:.2f}%)")
    print(f"- correct: {report['correct']}")
    print(f"- accuracy: {report['accuracy']:.2f}%")
    print(f"- mismatch: {report['mismatch']} ({report['mismatch_ratio']:.2f}%)")

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"- report: {report_path}")


if __name__ == "__main__":
    main()