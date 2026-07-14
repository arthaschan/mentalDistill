#!/usr/bin/env python3
"""Assemble the Choice-Head training file with DeepSeek-V3 soft labels (AIEA recipe).
Steps:
  1. Load DeepSeek hard labels for UK/US (from screening) + MedMCQA train.
  2. Map onto train_main by uid.
  3. hard->soft smoothing (smooth_eps=0.25: correct 0.8, others 0.05) as TeacherDist.
  4. Emit Question/Options/Answer(=GT)/TeacherDist for the trainer.
Also emits val/test files in the same schema (no teacher dist needed for eval).
"""
import json, os, math
OPT=["A","B","C","D","E"]
DS="english/dataset"; LP="english/01_teacher_screening/logprobs"
OUTD="english/03_main_distill/data"; os.makedirs(OUTD,exist_ok=True)
SMOOTH=0.25

def load(p): return [json.loads(l) for l in open(p)] if os.path.exists(p) else []
def hard_to_soft(ans,n_opts):
    letters=OPT[:n_opts] if n_opts in (4,5) else OPT
    d={}
    for k in letters:
        d[k]= (1.0-SMOOTH+SMOOTH/len(letters)) if k==ans else (SMOOTH/len(letters))
    return d

# collect DeepSeek hard labels by uid
ds_label={}
for f in ["DeepSeekV3_labels.jsonl","DeepSeekV3_medmcqa_train.jsonl","DeepSeekV3_medmcqa_gap.jsonl"]:
    for r in load(f"{LP}/{f}"):
        uid=r.get("uid"); ta=str(r.get("TeacherAnswer","")).strip().upper()
        if uid and ta in OPT: ds_label[uid]=ta

train=load(f"{DS}/train_main.jsonl")
miss=0; out=[]
for r in train:
    uid=r["uid"]; gt=r["answer"]; n=r["n_options"]
    ta=ds_label.get(uid)
    if ta is None: miss+=1; ta=gt   # fallback: no teacher label -> GT (rare)
    row={"uid":uid,"Question":r["stem"],"Options":r["options"],
         "Answer":gt,  # trainer's CE target = ground truth
         "TeacherAnswer":ta,"TeacherDist":hard_to_soft(ta,n),
         "group":r["group"],"n_options":n}
    out.append(row)
with open(f"{OUTD}/train_head_distill.jsonl","w") as w:
    for r in out: w.write(json.dumps(r,ensure_ascii=False)+"\n")

# eval files (GT only)
def emit_eval(src,dst):
    rows=load(src); o=[]
    for r in rows:
        o.append({"uid":r["uid"],"Question":r["stem"],"Options":r["options"],
                  "Answer":r["answer"],"group":r.get("group",""),"n_options":r["n_options"]})
    with open(dst,"w") as w:
        for r in o: w.write(json.dumps(r,ensure_ascii=False)+"\n")
    return len(o)
nv=emit_eval(f"{DS}/val.jsonl",f"{OUTD}/val.jsonl")
nt=emit_eval(f"{DS}/test_ukus.jsonl",f"{OUTD}/test_ukus.jsonl")
nm=emit_eval(f"{DS}/test_medmcqa.jsonl",f"{OUTD}/test_medmcqa.jsonl")

# teacher coverage / accuracy on train (sanity)
cov=sum(1 for r in out if r["TeacherAnswer"]==ds_label.get(r["uid"]))
teacher_acc=100*sum(1 for r in out if r["TeacherAnswer"]==r["Answer"])/len(out)
print(f"train_head_distill: {len(out)}  (teacher-label missing fallback={miss})")
print(f"DeepSeek teacher acc on train pool: {teacher_acc:.2f}%")
print(f"val={nv} test_ukus={nt} test_medmcqa={nm}")
print(f"-> {OUTD}/")
