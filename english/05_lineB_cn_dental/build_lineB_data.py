#!/usr/bin/env python3
"""Paper Line B: assemble Chinese dental-specialist training data.
Teacher = DeepSeek-V3 (Chinese, 86.4% on these). alpha=0 headline (per ablation), so trainer
uses GT as CE target; TeacherDist smoothed for the alpha>0 comparison arms.
Eval anchors: test_dental (125, the subset the original paper did NOT beat teacher on),
val_dental (125) for selection.
"""
import json, os, re
OPT=["A","B","C","D","E"]; SMOOTH=0.25
SRC="english/00_data/cmexam_dental_trainB_deepseek.jsonl"
OUTD="english/05_lineB_cn_dental/data"; os.makedirs(OUTD,exist_ok=True)

def hard_to_soft(ans):
    d={}
    for k in OPT: d[k]=(1.0-SMOOTH+SMOOTH/len(OPT)) if k==ans else (SMOOTH/len(OPT))
    return d
def norm(s): return re.sub(r'\s+',' ',str(s)).strip()

# training rows (single-answer only)
rows=[json.loads(l) for l in open(SRC)]
out=[]
for r in rows:
    gt=str(r.get("OriginalAnswer","")).upper()
    ta=str(r.get("TeacherAnswer","")).upper()
    if gt not in OPT or ta not in OPT: continue
    out.append({"uid":r.get("uid"),"Question":r["Question"],"Options":r["Options"],
                "Answer":gt,"TeacherAnswer":ta,"TeacherDist":hard_to_soft(ta)})
with open(f"{OUTD}/train_head_distill.jsonl","w") as w:
    for r in out: w.write(json.dumps(r,ensure_ascii=False)+"\n")

# eval files: copy the canonical 125 dental test + val from module 15
import shutil
for src,dst in [("15_fulldata_resplit/data/test_dental.jsonl",f"{OUTD}/test_dental.jsonl"),
                ("15_fulldata_resplit/data/val_dental.jsonl",f"{OUTD}/val_dental.jsonl")]:
    shutil.copy(src,dst)

teacher_acc=100*sum(1 for r in out if r["TeacherAnswer"]==r["Answer"])/len(out)
print(f"line-B train: {len(out)}  teacher acc={teacher_acc:.1f}%")
print(f"eval: test_dental=125, val_dental=125 (copied from module 15)")
print(f"-> {OUTD}/")
