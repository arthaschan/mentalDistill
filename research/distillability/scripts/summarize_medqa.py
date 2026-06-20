#!/usr/bin/env python3
"""汇总 MedQA 跨数据集蒸馏结果: 各教师 3-seed 三臂 test_acc, 算 geom-random / geom-baseline。
对比 CMExam 结果, 看"几何去噪增益"是否跨数据集成立。"""
import os, re, glob, statistics as st

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO = os.path.dirname(REPO)
RUNS = os.path.join(REPO, "research", "distillability", "runs")
TEST_RE = re.compile(r"test_acc=([0-9.]+)%")

def parse(log):
    if not os.path.exists(log): return None
    best = None
    for line in open(log, encoding="utf-8", errors="ignore"):
        if "[TEST-BEST]" in line:
            m = TEST_RE.search(line)
            if m: best = float(m.group(1))
    return best

print("="*70)
print("MedQA 跨数据集蒸馏结果 (3-seed)")
print("="*70)
print(f"{'teacher':10s}{'baseline':>10}{'geom':>9}{'random':>9}{'geom-base':>11}{'geom-rand':>11}")
cm_ref = {"Phi4":(1.65,0.81),"Yi34B":(1.41,0.60),"Qwen32B":None}  # CMExam参考(gain_base, geom-rand)
for teacher in ["Phi4","Yi34B","Qwen32B"]:
    arms = {"baseline_all":[], "geom_top50":[], "random_top50":[]}
    for arm in arms:
        for lg in glob.glob(os.path.join(RUNS, f"medqa_{teacher}", "logs", f"{arm}_seed*.log")):
            v = parse(lg)
            if v is not None: arms[arm].append(v)
    if not all(arms.values()):
        print(f"{teacher:10s}  [未完成: " + " ".join(f"{a}={len(v)}" for a,v in arms.items()) + "]")
        continue
    mb,mg,mr = st.mean(arms["baseline_all"]), st.mean(arms["geom_top50"]), st.mean(arms["random_top50"])
    print(f"{teacher:10s}{mb:>10.2f}{mg:>9.2f}{mr:>9.2f}{mg-mb:>+11.2f}{mg-mr:>+11.2f}")

print("\n判读: 若弱教师(Phi4/Yi34B)在MedQA上 geom-random 仍>0, 则'几何去噪增益'跨数据集成立。")
print("对比 CMExam: Phi4 geom-rand=+0.81, Yi34B=+0.60。看MedQA是否同向。")
