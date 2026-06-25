#!/usr/bin/env python3
"""
summarize_alpha_ablation.py — 汇总 α 消融结果, 按 val 均值选最优 α.

读取 runs/alpha_ablation_14b/logs/stage1_a*_s*.log, 解析每个 (α,seed) 的
val_acc 与 builtin test_acc, 按 α 聚合 3 种子 (mean/std/min/max),
按 **验证集均值** 选最优 α (绝不用 test 选点), 写出:
  - runs/alpha_ablation_14b/alpha_ablation_results.json
  - runs/alpha_ablation_14b/ALPHA_SUMMARY.md  (含曲线表 + 选点 + canonical-eval 命令)

注意: 这里的 acc 来自训练内置 prompt, 仅供 α 之间内部比较与选点.
头条数字须用 scripts/run_eval_dual.py 对选出的最优 α 重跑 (canonical prompt).
"""
import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

RUN_ROOT = Path(__file__).resolve().parent.parent / "runs" / "alpha_ablation_14b"
LOG_DIR = RUN_ROOT / "logs"
OUT_DIR = RUN_ROOT / "outputs"

VAL_RE = re.compile(r"\[VAL\] epoch=\d+ acc=([0-9.]+)")
TEST_RE = re.compile(r"\[TEST-BEST\] epoch=\d+ test_acc=([0-9.]+)")
NAME_RE = re.compile(r"stage1_a(?P<atag>[0-9p]+)_s(?P<seed>\d+)\.log$")


def atag_to_alpha(atag: str) -> float:
    return float(atag.replace("p", "."))


def parse_logs():
    rows = []
    if not LOG_DIR.is_dir():
        return rows
    for log in sorted(LOG_DIR.glob("stage1_a*_s*.log")):
        m = NAME_RE.search(log.name)
        if not m:
            continue
        alpha = atag_to_alpha(m.group("atag"))
        seed = int(m.group("seed"))
        text = log.read_text(errors="ignore")
        vals = VAL_RE.findall(text)
        tests = TEST_RE.findall(text)
        val_acc = float(vals[-1]) if vals else None
        test_acc = float(tests[-1]) if tests else None
        # 确认 adapter 真的落盘 (区分"跑完"与"中途挂")
        best_ok = (OUT_DIR / f"a{m.group('atag')}_s{seed}" /
                   "stage1_head" / "best" / "adapter_config.json").exists()
        rows.append({"alpha": alpha, "seed": seed, "val_acc": val_acc,
                     "test_acc_builtin": test_acc, "best_saved": best_ok})
    return rows


def agg(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return {
        "mean": round(st.mean(vals), 3),
        "std": round(st.pstdev(vals), 3) if len(vals) > 1 else 0.0,
        "min": min(vals), "max": max(vals), "n": len(vals),
    }


def main():
    rows = parse_logs()
    if not rows:
        print(f"[WARN] 未在 {LOG_DIR} 找到任何 stage1_a*_s*.log — 实验可能尚未开始.")
        return

    by_alpha = defaultdict(lambda: {"val": [], "test": [], "seeds": []})
    for r in rows:
        by_alpha[r["alpha"]]["val"].append(r["val_acc"])
        by_alpha[r["alpha"]]["test"].append(r["test_acc_builtin"])
        by_alpha[r["alpha"]]["seeds"].append(r["seed"])

    summary = []
    for alpha in sorted(by_alpha):
        d = by_alpha[alpha]
        summary.append({
            "alpha": alpha,
            "n_seeds": len([v for v in d["val"] if v is not None]),
            "val": agg(d["val"]),
            "test_builtin": agg(d["test"]),
        })

    # 按 val 均值选最优 (完整 3 种子的 α 才有资格)
    complete = [s for s in summary if s["val"] and s["val"]["n"] == 3]
    best = max(complete, key=lambda s: s["val"]["mean"]) if complete else None

    results = {"rows": rows, "by_alpha": summary,
               "best_alpha_by_val_mean": best["alpha"] if best else None}
    (RUN_ROOT / "alpha_ablation_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    # markdown 摘要
    lines = ["# α-Ablation Summary (14B, Stage-1, builtin-eval)", "",
             "Loss = α·KL + (1−α)·CE. 选点用 **val 均值**, 头条须另跑 canonical eval.", "",
             "| α | n | val mean±std | val [min,max] | test(builtin) mean±std |",
             "|---|---|--------------|---------------|------------------------|"]
    for s in summary:
        v, t = s["val"], s["test_builtin"]
        vstr = f"{v['mean']:.2f}±{v['std']:.2f}" if v else "NA"
        vrng = f"[{v['min']:.2f},{v['max']:.2f}]" if v else "NA"
        tstr = f"{t['mean']:.2f}±{t['std']:.2f}" if t else "NA"
        star = "  ⬅ best(val)" if best and s["alpha"] == best["alpha"] else ""
        lines.append(f"| {s['alpha']} | {s['n_seeds']} | {vstr} | {vrng} | {tstr}{star} |")

    lines += ["", "## 选点", ""]
    if best:
        lines.append(f"- **最优 α (按 val 均值) = {best['alpha']}**, "
                     f"val={best['val']['mean']:.2f}±{best['val']['std']:.2f}%")
        a0 = next((s for s in summary if s["alpha"] == 0.0 and s["val"]), None)
        if a0:
            delta = best["val"]["mean"] - a0["val"]["mean"]
            lines.append(f"- α=0 (纯CE/SFT) val={a0['val']['mean']:.2f}% → "
                         f"最优−α0 = **{delta:+.2f}pp** (val). "
                         f"{'教师KL有实质增量(≥1pp)' if delta >= 1.0 else '⚠️ 增量<1pp: 教师信号可能可有可无, 见方案第4节'}")
        lines += ["", "## 下一步: 对最优 α 做 canonical 评估 (对齐论文 Table I)", "",
                  "```bash", "cd /home/student/arthas/mentalDistill && source setup.env"]
        atag = str(best["alpha"]).replace(".", "p")
        lines.append(f"python3 15_fulldata_resplit/scripts/run_eval_dual.py \\")
        lines.append(f"    --run_root 15_fulldata_resplit/runs/alpha_ablation_14b \\")
        lines.append(f"    --student_size 14b")
        lines += ["```",
                  f"(将评估 α={best['alpha']} 的 3 种子在 full 991 + dental 125 上的 canonical 准确率, "
                  "与教师 87.18% / 基线 83.55% / 主结果 88.67% 同口径比较.)"]
    else:
        lines.append("- 尚无任何 α 完成全部 3 种子, 暂不能选点.")

    (RUN_ROOT / "ALPHA_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[OK] 写出 {RUN_ROOT/'alpha_ablation_results.json'}")
    print(f"[OK] 写出 {RUN_ROOT/'ALPHA_SUMMARY.md'}")


if __name__ == "__main__":
    main()
