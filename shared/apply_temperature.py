#!/usr/bin/env python3
"""
apply_temperature.py — 对融合软标签施加温度缩放

p_T(x) = p(x)^(1/T) / Z, 其中 Z = Σ p(x)^(1/T)
- T=1: 不变
- T>1: 更平滑 (更均匀)
- T<1: 更尖锐 (更接近 argmax)
"""
import argparse
import json
import math
from pathlib import Path

OPTION_LETTERS = ["A", "B", "C", "D", "E"]


def apply_temperature(dist: dict, T: float) -> dict:
    if T == 1.0:
        return dist
    log_probs = {}
    for ch in OPTION_LETTERS:
        p = max(float(dist.get(ch, 0.0)), 1e-12)
        log_probs[ch] = math.log(p) / T
    # softmax for numerical stability
    max_lp = max(log_probs.values())
    exp_vals = {ch: math.exp(log_probs[ch] - max_lp) for ch in OPTION_LETTERS}
    Z = sum(exp_vals.values())
    return {ch: exp_vals[ch] / Z for ch in OPTION_LETTERS}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--temperature", type=float, required=True)
    args = p.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with open(args.input, "r", encoding="utf-8") as rf, \
         open(args.output, "w", encoding="utf-8") as wf:
        for line in rf:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            dist = row.get("TeacherDist", {})
            if dist:
                row["TeacherDist"] = apply_temperature(dist, args.temperature)
            wf.write(json.dumps(row, ensure_ascii=False) + "\n")
            total += 1

    print(f"[TEMP] T={args.temperature}, processed {total} samples → {args.output}")


if __name__ == "__main__":
    main()
