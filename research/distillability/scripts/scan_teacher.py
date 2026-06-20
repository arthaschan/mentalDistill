#!/usr/bin/env python3
"""
scan_teacher.py — 教师模型"可蒸馏性体检"工具 (一键: 喂模型 → 出报告 → 出选择性蒸馏数据)

核心价值(非"按熵过滤"换皮):
  1. label-free 标记不可信样本 (纯熵, 无需训练/无需对工具调参)
  2. 领域/难度地图: 指出教师在哪些子领域不可信 (人能看懂)
  3. 自证模块: 若数据带 Difficulty level, 自动算"不可信↔人类难度"相关, 报告检测可信度
     (这是相对置信度校准方法的护城河: 证明不确定=客观难点)
  4. 产出已剔除不可信样本的选择性蒸馏数据集

未来用法(验证泛化): 对未参与研究的新模型(DeepSeek/Mistral等)跑本工具,
  若同样能定位不可信区且与人类难度正相关 → 证明工具泛化、规律普适。

用法:
  # 模式1: 已有 logprobs jsonl, 直接体检
  python scan_teacher.py --logprobs path/to/teacher_logprobs.jsonl --label MyModel --out_dir reports/

  # 模式2: 从模型现场生成 logprobs 再体检
  python scan_teacher.py --model_path models/xxx --dataset data/train.jsonl --label MyModel --out_dir reports/
"""
import argparse, json, os, subprocess, sys
import numpy as np

OPTION_LETTERS = ["A", "B", "C", "D", "E"]
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO = os.path.dirname(REPO)
SHARED = os.path.join(REPO, "shared")


def entropy_margin_peak(raw):
    p = np.clip(np.array(raw, dtype=np.float64), 1e-12, None); p = p / p.sum()
    srt = np.sort(p)[::-1]
    ent = float(-np.sum(p * np.log(p)))
    return ent, float(srt[0] - srt[1]), float(srt[0])


def load_rows(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        dist = r.get("TeacherDist", {})
        gt = str(r.get("OriginalAnswer") or r.get("Answer", "")).strip().upper()
        if not dist or gt not in OPTION_LETTERS:
            continue
        raw = [float(dist.get(c, 0.0)) for c in OPTION_LETTERS]
        if sum(raw) <= 1e-9:
            continue
        ta = OPTION_LETTERS[int(np.argmax(raw))]
        ent, margin, peak = entropy_margin_peak(raw)
        rows.append({
            "ent": ent, "margin": margin, "peak": peak,
            "correct": 1 if ta == gt else 0,
            "difficulty": str(r.get("Difficulty level", "")).strip(),
            "domain": str(r.get("Medical Discipline", "")).strip() or "unknown",
        })
    return rows


def maybe_generate_logprobs(model_path, dataset, label, out_dir):
    """模式2: 调用现有脚本生成 logprobs。"""
    lp = os.path.join(out_dir, f"{label}_logprobs.jsonl")
    if os.path.exists(lp) and sum(1 for _ in open(lp)) > 50:
        print(f"[复用] 已存在 logprobs: {lp}")
        return lp
    print(f"[生成] {label} logprobs (调用 generate_teacher_labels_local_logprobs.py)...")
    cmd = [sys.executable, os.path.join(SHARED, "generate_teacher_labels_local_logprobs.py"),
           "--model_path", model_path, "--dataset", dataset,
           "--output", lp, "--gt_field", "Answer", "--resume"]
    subprocess.run(cmd, check=True)
    return lp


def spearman(a, b):
    try:
        from scipy import stats
        r = stats.spearmanr(a, b)
        return float(r.correlation), float(r.pvalue)
    except Exception:
        a = np.array(a); b = np.array(b)
        ar = np.argsort(np.argsort(a)); br = np.argsort(np.argsort(b))
        if ar.std() == 0 or br.std() == 0:
            return 0.0, 1.0
        return float(np.corrcoef(ar, br)[0, 1]), float("nan")


def main():
    ap = argparse.ArgumentParser(description="教师可蒸馏性体检工具")
    ap.add_argument("--logprobs", help="已有 teacher logprobs jsonl (模式1)")
    ap.add_argument("--model_path", help="模型路径 (模式2, 现场生成 logprobs)")
    ap.add_argument("--dataset", help="带GT的探针数据集 jsonl (模式2)")
    ap.add_argument("--label", required=True, help="模型名(报告标识)")
    ap.add_argument("--out_dir", default="research/distillability/reports")
    ap.add_argument("--keep_frac", type=float, default=0.5, help="选择性蒸馏保留比例(按可信度)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 取得 logprobs
    if args.logprobs:
        lp = args.logprobs
    elif args.model_path and args.dataset:
        lp = maybe_generate_logprobs(args.model_path, args.dataset, args.label, args.out_dir)
    else:
        ap.error("需提供 --logprobs 或 (--model_path 且 --dataset)")

    rows = load_rows(lp)
    if not rows:
        print("[错误] 无有效样本"); return
    n = len(rows)
    y = np.array([r["correct"] for r in rows])
    ent = np.array([r["ent"] for r in rows])
    acc = float(y.mean() * 100)

    report = {"label": args.label, "n_samples": n, "overall_acc": round(acc, 2),
              "mean_entropy": round(float(ent.mean()), 4)}

    # 1. 样本级不可信标记 (label-free: 纯熵, 越高越不可信)
    thr = np.quantile(ent, args.keep_frac)   # 保留熵最低(最可信)的 keep_frac
    untrust = ent > thr
    report["untrusted_fraction"] = round(float(untrust.mean()), 4)
    report["untrusted_error_rate"] = round(float((1 - y[untrust]).mean() * 100), 2) if untrust.sum() else None
    report["trusted_error_rate"] = round(float((1 - y[~untrust]).mean() * 100), 2) if (~untrust).sum() else None

    # 2. 难度自证 (护城河)
    diffs = [int(r["difficulty"]) for r in rows if r["difficulty"] in ["1","2","3","4","5"]]
    if len(diffs) > 100:
        ent_d = [r["ent"] for r in rows if r["difficulty"] in ["1","2","3","4","5"]]
        rho, p = spearman(ent_d, diffs)
        report["self_validation"] = {
            "entropy_vs_human_difficulty_spearman": round(rho, 4),
            "p_value": p,
            "verdict": ("可信:不确定区对应人类真实难度" if rho > 0.2
                        else "弱:与人类难度关联不强,慎用" if rho > 0.1
                        else "存疑:不确定区与人类难度无关")
        }

    # 3. 领域地图
    domains = {}
    for r in rows:
        domains.setdefault(r["domain"], []).append(r["correct"])
    dmap = {}
    for d, cs in domains.items():
        if len(cs) < 20:
            continue
        da = float(np.mean(cs) * 100)
        dmap[d] = {"n": len(cs), "acc": round(da, 2), "delta_vs_overall": round(da - acc, 2)}
    report["domain_map"] = dict(sorted(dmap.items(), key=lambda x: x[1]["delta_vs_overall"]))

    # 4. 产出选择性蒸馏数据集(剔除不可信样本) —— 索引列表
    keep_idx = [i for i in range(n) if not untrust[i]]
    report["selective_distill"] = {"keep_frac": args.keep_frac, "kept": len(keep_idx),
                                    "dropped": int(untrust.sum())}

    # 写报告
    rpath = os.path.join(args.out_dir, f"{args.label}_health_report.json")
    json.dump(report, open(rpath, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # 可读 markdown
    md = [f"# {args.label} 可蒸馏性体检报告\n",
          f"- 样本数: {n}", f"- 整体准确率: {acc:.1f}%", f"- 平均熵: {ent.mean():.4f}",
          f"- 不可信样本占比: {report['untrusted_fraction']*100:.0f}% "
          f"(错误率 {report['untrusted_error_rate']}% vs 可信区 {report['trusted_error_rate']}%)\n"]
    if "self_validation" in report:
        sv = report["self_validation"]
        md.append(f"## 自证 (不可信↔人类难度)\n- Spearman ρ={sv['entropy_vs_human_difficulty_spearman']} "
                  f"→ {sv['verdict']}\n")
    md.append("## 领域地图 (负值=该领域不如整体, 越负越不该信)\n")
    md.append("| 领域 | 样本 | 准确率 | vs整体 |\n|---|---|---|---|")
    for d, v in report["domain_map"].items():
        flag = " ⚠️" if v["delta_vs_overall"] < -5 else ""
        md.append(f"| {d} | {v['n']} | {v['acc']}% | {v['delta_vs_overall']:+.1f}{flag} |")
    mdpath = os.path.join(args.out_dir, f"{args.label}_health_report.md")
    open(mdpath, "w", encoding="utf-8").write("\n".join(md))

    print(f"[报告] {rpath}")
    print(f"[可读] {mdpath}")
    print(f"  整体 {acc:.1f}% | 不可信区错误率 {report['untrusted_error_rate']}% vs 可信区 {report['trusted_error_rate']}%")
    if "self_validation" in report:
        print(f"  自证: {report['self_validation']['verdict']} (ρ={report['self_validation']['entropy_vs_human_difficulty_spearman']})")


if __name__ == "__main__":
    main()
