#!/usr/bin/env python3
"""
D2: 拟合"学生参数量 vs 蒸馏后正确率"曲线, 反查容量下限。

读各尺寸学生的 test_acc, 拟合 acc ~ a*log(params)+b, 给定目标正确率反查最小容量。
纯 CPU。
"""
import os, re, json, math

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO = os.path.dirname(REPO)
RUNS = os.path.join(REPO, "research", "distillability", "runs", "d2_capacity")
OUT = os.path.join(REPO, "research", "distillability", "outputs", "d2_capacity_curve.json")
TEST_RE = re.compile(r"test_acc=([0-9.]+)%")

SIZES = {"0.5B": 0.5, "1.5B": 1.5, "3B": 3.0, "7B": 7.0, "14B": 14.0}


def parse(size):
    log = os.path.join(RUNS, f"qwen{size}_seed42", "stage1_head")
    # 实际日志在 d2_size_<size>.log
    alt = os.path.join(REPO, "research", "distillability", f"d2_size_{size}.log")
    best = None
    for path in [alt]:
        if os.path.exists(path):
            for line in open(path, encoding="utf-8", errors="ignore"):
                if "[TEST-BEST]" in line:
                    m = TEST_RE.search(line)
                    if m:
                        best = float(m.group(1))
    return best


def linfit(xs, ys):
    """最小二乘拟合 y = a*x + b。"""
    n = len(xs)
    mx = sum(xs) / n; my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    a = num / den if den else 0
    b = my - a * mx
    return a, b


def main():
    pts = []
    for size, b in SIZES.items():
        acc = parse(size)
        if acc is not None:
            pts.append((b, acc))
            print(f"  Qwen2.5-{size:5s} ({b:>4}B params): test_acc={acc:.2f}%")
    if len(pts) < 2:
        print(f"[等待] 已完成 {len(pts)} 个尺寸, 不足以拟合曲线。")
        return

    xs = [math.log10(p[0]) for p in pts]
    ys = [p[1] for p in pts]
    a, b = linfit(xs, ys)
    print(f"\n拟合: acc ≈ {a:.2f}*log10(params_B) + {b:.2f}")

    # 反查: 给定目标正确率, 求最小 params
    res = {"points": [{"params_B": p[0], "acc": p[1]} for p in pts],
           "fit": {"slope_a": round(a, 3), "intercept_b": round(b, 3)},
           "capacity_lower_bound": {}}
    print("\n容量下限预测 (达到目标正确率的最小学生):")
    for target in [80, 83, 85, 86, 87]:
        # target = a*log10(P)+b -> P = 10^((target-b)/a)
        if a > 0:
            P = 10 ** ((target - b) / a)
            res["capacity_lower_bound"][f"{target}%"] = round(P, 2)
            print(f"  达到 {target}%: 需 ≥ {P:.2f}B 参数")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n[SAVED] {OUT}")
    print("\n注: log线性是简化模型; 若曲线饱和(大模型边际收益递减), 需换饱和函数拟合。")
    print("    外推到训练范围外不可靠, 仅内插可信。")


if __name__ == "__main__":
    main()
