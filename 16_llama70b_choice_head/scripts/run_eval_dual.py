#!/usr/bin/env python3
"""
Dual evaluation: test on full test set (991) + dental subset (125).
Evaluates all adapters found in a run directory.

Usage:
    python scripts/run_eval_dual.py --run_root runs/XXXX --student_size 14b
"""
import argparse
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


def find_adapters(run_root):
    adapters = []
    for root, dirs, files in os.walk(run_root / "outputs"):
        if "best" in dirs:
            best_dir = Path(root) / "best"
            if (best_dir / "adapter_config.json").exists():
                parent = best_dir.parent.name
                config_name = best_dir.parent.parent.name
                adapters.append({
                    "path": str(best_dir),
                    "stage": parent,
                    "config": config_name,
                    "label": f"{config_name}/{parent}",
                })
    return sorted(adapters, key=lambda x: x["label"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_root", required=True)
    p.add_argument("--student_size", required=True, choices=["7b", "14b"])
    p.add_argument("--data_dir", default=None)
    args = p.parse_args()

    root = Path(__file__).resolve().parent.parent
    shared_dir = root.parent / "shared"
    run_root = Path(args.run_root)

    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = root.parent / "15_fulldata_resplit" / "data"

    test_full = data_dir / "test.jsonl"
    test_dental = data_dir / "test_dental.jsonl"

    env_var = "BASE_MODEL_7B" if args.student_size == "7b" else "BASE_MODEL_14B"
    model_name = "Qwen2.5-7B-Instruct" if args.student_size == "7b" else "Qwen2.5-14B-Instruct"

    base_model = os.environ.get(env_var, "")
    if not base_model:
        candidate = root.parent / "models" / model_name
        if candidate.exists():
            base_model = str(candidate)
        else:
            print(f"ERROR: Set {env_var} or place model in models/{model_name}")
            sys.exit(1)

    adapters = find_adapters(run_root)
    if not adapters:
        print(f"No adapters found in {run_root}/outputs")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Dual Evaluation: {args.student_size.upper()} student (Llama-70B teacher)")
    print(f"Base model: {base_model}")
    print(f"Full test: {test_full} ({sum(1 for _ in open(test_full))} questions)")
    print(f"Dental test: {test_dental} ({sum(1 for _ in open(test_dental))} questions)")
    print(f"Adapters found: {len(adapters)}")
    print(f"{'='*60}\n")

    eval_script = shared_dir / "evaluate_model.py"
    results = []

    for adapter in adapters:
        print(f"\n--- Evaluating: {adapter['label']} ---")

        for test_name, test_file in [("full", test_full), ("dental", test_dental)]:
            wrong_log = run_root / "outputs" / f"wrong_{adapter['config']}_{adapter['stage']}_{test_name}.jsonl"

            cmd = [
                sys.executable, str(eval_script),
                "--base_model", base_model,
                "--adapter_dir", adapter["path"],
                "--test_data", str(test_file),
                "--wrong_log", str(wrong_log),
            ]

            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                output = result.stdout + result.stderr

                acc = None
                for line in output.split("\n"):
                    if "accuracy" in line.lower() or "正确率" in line.lower() or "准确率" in line.lower() or "correct" in line.lower():
                        m = re.search(r'(\d+\.?\d*)\s*%', line)
                        if m:
                            acc = float(m.group(1))
                        else:
                            m = re.search(r'(\d+)\s*/\s*(\d+)', line)
                            if m:
                                acc = 100.0 * int(m.group(1)) / int(m.group(2))
                            else:
                                m = re.search(r'[:：]\s*(\d+\.\d+)', line)
                                if m:
                                    acc = float(m.group(1))

                if acc is not None:
                    print(f"  [{test_name:>6}] {acc:.2f}%")
                else:
                    print(f"  [{test_name:>6}] (parse failed)")
                    print(f"  Output: {output[:200]}")

                results.append({
                    "config": adapter["config"],
                    "stage": adapter["stage"],
                    "test_set": test_name,
                    "accuracy": acc,
                    "adapter_path": adapter["path"],
                })
            except Exception as e:
                print(f"  [{test_name:>6}] ERROR: {e}")
                results.append({
                    "config": adapter["config"],
                    "stage": adapter["stage"],
                    "test_set": test_name,
                    "accuracy": None,
                    "error": str(e),
                })

    results_path = run_root / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n{'='*60}")
    print(f"Results saved to {results_path}")

    print(f"\n{'Config':<40} {'Stage':<15} {'Full%':>8} {'Dental%':>8}")
    print("-" * 75)

    grouped = defaultdict(dict)
    for r in results:
        key = (r["config"], r["stage"])
        grouped[key][r["test_set"]] = r.get("accuracy")

    for (config, stage), accs in sorted(grouped.items()):
        full_acc = f"{accs.get('full', 0):.2f}" if accs.get('full') is not None else "N/A"
        dental_acc = f"{accs.get('dental', 0):.2f}" if accs.get('dental') is not None else "N/A"
        print(f"{config:<40} {stage:<15} {full_acc:>8} {dental_acc:>8}")

    print(f"{'='*60}")


if __name__ == "__main__":
    main()
