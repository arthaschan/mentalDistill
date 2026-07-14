#!/usr/bin/env python3
"""Aggregate main distillation results from run logs into a results table."""
import re, os, glob, json
import numpy as np

RUN="english/03_main_distill/runs"
TEACHER_ACC=82.86  # DeepSeek-V3 on English dental (screening)

def grab(path, patterns):
    if not os.path.exists(path): return None
    txt=open(path,encoding="utf-8",errors="ignore").read()
    for pat in patterns:
        m=re.findall(pat,txt)
        if m: return float(m[-1])
    return None

# zero-shot floor
zs={}
for size in ["7B","14B"]:
    v=grab(f"{RUN}/zeroshot_{size}_ukus.log",[r'准确率[:：]\s*([\d.]+)%',r'accuracy[:：]?\s*([\d.]+)%',r'acc[=:]\s*([\d.]+)%'])
    zs[size]=v

ARM_ALPHA={"a00":0.0,"a35":0.35,"a10":1.0}
rows=[]
for d in sorted(glob.glob(f"{RUN}/*_s*")):
    name=os.path.basename(d)
    m=re.match(r'(7B|14B)_(a00|a35|a10)_s(\d+)',name)
    if not m: continue
    size,arm,seed=m.group(1),m.group(2),int(m.group(3))
    ukus=grab(f"{d}/train.log",[r'\[TEST-BEST\][^\n]*test_acc=([\d.]+)%',r'测试集准确率[:：]\s*([\d.]+)%'])
    val =grab(f"{d}/train.log",[r'\[BEST\] val_acc=([\d.]+)%',r'\[VAL\][^\n]*acc=([\d.]+)%'])
    med =grab(f"{d}/eval_medmcqa.log",[r'准确率[:：]\s*([\d.]+)%',r'accuracy[:：]?\s*([\d.]+)%',r'acc[=:]\s*([\d.]+)%'])
    rows.append(dict(size=size,arm=arm,seed=seed,val=val,test_ukus=ukus,test_medmcqa=med))

def agg(size,arm,key):
    vs=[r[key] for r in rows if r["size"]==size and r["arm"]==arm and r[key] is not None]
    return (round(np.mean(vs),2),round(np.std(vs),2),len(vs)) if vs else (None,None,0)

out={"teacher_deepseek_v3":TEACHER_ACC,"zeroshot_floor":zs,"runs":rows,"summary":{}}
lines=["# English Dental Distillation — Main Results\n",
       f"Teacher DeepSeek-V3: **{TEACHER_ACC}%** (English dental screening)\n",
       f"Zero-shot student floor (test_ukus): 7B={zs.get('7B')}%  14B={zs.get('14B')}%\n",
       "## Distilled vs GT-only (mean±std over seeds)",
       "| student | arm | test_ukus (primary) | test_medmcqa (cross-src) | val |",
       "|---|---|---|---|---|"]
for size in ["7B","14B"]:
    for arm in ["a00","a35","a10"]:
        u=agg(size,arm,"test_ukus"); mm=agg(size,arm,"test_medmcqa"); v=agg(size,arm,"val")
        out["summary"][f"{size}_{arm}"]={"alpha":ARM_ALPHA[arm],"test_ukus":u,"test_medmcqa":mm,"val":v}
        lines.append(f"| {size} | α={ARM_ALPHA[arm]} | {u[0]}±{u[1]} (n={u[2]}) | {mm[0]}±{mm[1]} | {v[0]}±{v[1]} |")
# headline checks: does the Chinese finding (alpha=0 best, KL hurts) replicate?
lines.append("\n## Headline checks (replication of Chinese α ablation)")
for size in ["7B","14B"]:
    a0=out["summary"][f"{size}_a00"]["test_ukus"][0]
    a35=out["summary"][f"{size}_a35"]["test_ukus"][0]
    a10=out["summary"][f"{size}_a10"]["test_ukus"][0]
    if a0 is not None and a35 is not None:
        lines.append(f"- {size}: α=0 {a0}% vs α=0.35 {a35}% = **{a0-a35:+.2f}pp** (KL权重的影响)")
    if a0 is not None and a10 is not None:
        lines.append(f"- {size}: α=0 {a0}% vs α=1.0(纯KL) {a10}% = **{a0-a10:+.2f}pp** (纯模仿教师代价)")
    if a0 is not None:
        lines.append(f"- {size}: α=0 {a0}% vs teacher {TEACHER_ACC}% = {a0-TEACHER_ACC:+.2f}pp (student vs teacher)")
    mono = (a0 is not None and a35 is not None and a10 is not None and a0>=a35>=a10)
    if a0 is not None and a35 is not None and a10 is not None:
        lines.append(f"- {size}: 单调性 α0≥α0.35≥α1.0 = **{mono}** ({'复制中文结论' if mono else '未完全单调,如实报告'})")
open(f"{RUN}/RESULTS.md","w").write("\n".join(lines))
json.dump(out,open(f"{RUN}/results.json","w"),ensure_ascii=False,indent=2)
print("\n".join(lines))
print(f"\n-> {RUN}/RESULTS.md")
