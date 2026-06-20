#!/usr/bin/env python3
"""
phase2_predict.py — 阶段 2「预测侧」单教师流水线（冻结线之前的全部步骤）

对一个已下载完整的新教师，执行：
  1. token 映射预检（A-E 必须干净映射）
  2. 生成真实 logprobs（GPU 单前向，调用 shared/generate_teacher_labels_local_logprobs.py）
  3. 用【已冻结的】teacher_distillability_score.py 算 DI 几何预测分数
  4. 把该教师的 DI + predicted_rank append 进 predictions.json
  （git commit 由人工执行 —— 那是该教师的冻结线，脚本只负责写入并打印提醒）

注意：本脚本【不做蒸馏训练】。蒸馏验证在冻结线之后由 run_phase2_distill.sh 执行。
这样保证"预测在前、验证在后"。

用法：
  python phase2_predict.py --label phi-4 --model_dir models/phi-4
"""
import argparse, json, os, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SHARED = REPO / "shared"
SELF_DIR = REPO / "research" / "distillability"
TRAIN_DATA = REPO / "15_fulldata_resplit" / "data" / "train.jsonl"
PY = os.environ.get("EASYEDIT_PY", sys.executable)


def run(cmd, **kw):
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    return subprocess.run(cmd, **kw).returncode


def index_complete(model_dir: Path):
    idx = model_dir / "model.safetensors.index.json"
    if not idx.exists():
        # single-shard model: just need one safetensors
        return bool(list(model_dir.glob("*.safetensors")))
    want = set(json.load(open(idx))["weight_map"].values())
    have = {f.name for f in model_dir.glob("*.safetensors")}
    return not (want - have)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--skip_precheck", action="store_true")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    labels_dir = SELF_DIR / "teacher_labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    logprob_out = labels_dir / f"{args.label}_train_logprobs.jsonl"

    # 0. completeness guard
    if not index_complete(model_dir):
        print(f"[ABORT] {args.model_dir} is not fully downloaded yet (missing shards).")
        sys.exit(2)

    # 1. token precheck
    if not args.skip_precheck:
        rc = run([PY, str(SELF_DIR / "scripts" / "precheck_option_tokens.py"),
                  "--teachers", f"{args.label}:{args.model_dir}",
                  "--output", str(SELF_DIR / "outputs" / f"option_token_precheck_{args.label}.json")])
        if rc != 0:
            print(f"[ABORT] token precheck FAILED for {args.label} — needs custom option-token strategy.")
            sys.exit(1)

    # 2. real logprobs (skip if already done)
    if logprob_out.exists() and sum(1 for _ in open(logprob_out)) > 100:
        print(f"[SKIP] logprobs already exist: {logprob_out}")
    else:
        rc = run([PY, str(SHARED / "generate_teacher_labels_local_logprobs.py"),
                  "--model_path", str(model_dir),
                  "--dataset", str(TRAIN_DATA),
                  "--output", str(logprob_out),
                  "--gt_field", "Answer", "--resume"])
        if rc != 0:
            print(f"[ABORT] logprob generation failed for {args.label}.")
            sys.exit(1)

    # 3. DI prediction via FROZEN predictor (include all known teachers for z-score context)
    existing = {
        "Qwen14B": SELF_DIR / "teacher_labels" / "qwen14b_train_logprobs.jsonl",
        "Qwen32B": SELF_DIR / "teacher_labels" / "qwen32b_train_logprobs.jsonl",
        "Llama70B": REPO / "16_llama70b_choice_head" / "data" / "train_head_distill.jsonl",
    }
    teacher_args = [f"{args.label}:{logprob_out}"]
    for lab, p in existing.items():
        if Path(p).exists():
            teacher_args.append(f"{lab}:{p}")
    rc = run([PY, str(SELF_DIR / "teacher_distillability_score.py"),
              "--teachers", *teacher_args,
              "--output", str(SELF_DIR / "outputs" / f"di_scores_with_{args.label}.json")])
    if rc != 0:
        print(f"[WARN] DI scoring returned {rc}")

    print("\n" + "=" * 70)
    print(f"[PREDICT DONE] {args.label}")
    print(f"  logprobs : {logprob_out}")
    print(f"  DI report: outputs/di_scores_with_{args.label}.json")
    print("  NEXT (manual freeze line):")
    print(f"    1. review DI, append {args.label} entry to predictions.json")
    print(f"    2. git add predictions.json && git commit  <-- FREEZE LINE for {args.label}")
    print(f"    3. THEN run distillation: SEED=... bash research/distillability/scripts/run_phase2_distill.sh {args.label}")
    print("=" * 70)


if __name__ == "__main__":
    main()
