#!/usr/bin/env python3
"""Fusion oracle upper bound (Screening #3, GO/NO-GO gate) — ZERO training.
Reuses screening logprobs. Computes, per item, several label-fusion strategies and
their accuracy ceiling vs the best single teacher. Verdict thresholds (pre-registered):
  ceiling - best_single >= 2.0pp  -> GO  (fuse; path 3a, cross-lingual story)
           < 0.5pp                -> NO-GO (single teacher; path 3b)
  in between                      -> WEAK-GO (record, secondary experiment)

Strategies:
  best_single      : the single best teacher overall (baseline to beat)
  oracle_anyright  : upper ceiling — correct if ANY teacher is right (loose bound)
  majority_vote    : hard-label plurality across teachers
  conf_route       : per item, take the label of the teacher with LOWEST entropy (most confident)
  domain_route     : per item, take the label of that subject's best-overall teacher (from screening)
  prob_avg         : average the ABCDE prob vectors across teachers, argmax
"""
import json, os, glob
import numpy as np
from collections import defaultdict, Counter

LP="english/01_teacher_screening/logprobs"
REP="english/02_fusion_oracle"; os.makedirs(REP,exist_ok=True)
LETTERS=["A","B","C","D","E"]

def load(path):
    out={}
    for line in open(path):
        line=line.strip()
        if not line: continue
        try: r=json.loads(line)
        except: continue
        dist=r.get("TeacherDist",{})
        gt=str(r.get("OriginalAnswer") or r.get("Answer","")).strip().upper()
        uid=r.get("uid")
        if not dist or gt not in LETTERS or not uid: continue
        raw=np.array([float(dist.get(c,0.0)) for c in LETTERS])
        if raw.sum()<=1e-9: continue
        raw=raw/raw.sum()
        ent=float(-np.sum(np.clip(raw,1e-12,None)*np.log(np.clip(raw,1e-12,None))))
        out[uid]={"gt":gt,"p":raw,"pred":LETTERS[int(np.argmax(raw))],"ent":ent,
                  "subj":r.get("Medical Discipline","?")}
    return out

teachers={}
for f in sorted(glob.glob(f"{LP}/*_logprobs.jsonl")):
    name=os.path.basename(f).replace("_logprobs.jsonl","")
    d=load(f)
    if d: teachers[name]=d
names=list(teachers)

# common uids across all teachers
common=set.intersection(*[set(d) for d in teachers.values()])
common=sorted(common)
N=len(common)

# per-teacher overall acc (on common set) + subject-best map
overall={n:100*np.mean([teachers[n][u]["gt"]==teachers[n][u]["pred"] for u in common]) for n in names}
best_single=max(overall,key=overall.get)
# subject -> best overall teacher restricted by per-subject acc
subj_items=defaultdict(list)
for u in common: subj_items[teachers[names[0]][u]["subj"]].append(u)
subj_best={}
for s,us in subj_items.items():
    accs={n:np.mean([teachers[n][u]["gt"]==teachers[n][u]["pred"] for u in us]) for n in names}
    subj_best[s]=max(accs,key=accs.get)

def acc_of(pred_fn):
    c=0
    for u in common:
        if pred_fn(u)==teachers[names[0]][u]["gt"]: c+=1
    return 100*c/N

def gt(u): return teachers[names[0]][u]["gt"]

res={}
res["best_single_"+best_single]=overall[best_single]
# oracle any-right (loose ceiling)
res["oracle_anyright"]=100*np.mean([any(teachers[n][u]["pred"]==gt(u) for n in names) for u in common])
# majority vote
def majvote(u):
    votes=Counter(teachers[n][u]["pred"] for n in names)
    top=max(votes.values()); cands=[k for k,v in votes.items() if v==top]
    if len(cands)==1: return cands[0]
    # tie-break by summed prob
    s={c:sum(teachers[n][u]["p"][LETTERS.index(c)] for n in names) for c in cands}
    return max(s,key=s.get)
res["majority_vote"]=acc_of(majvote)
# confidence route (lowest entropy teacher wins)
res["conf_route"]=acc_of(lambda u: teachers[min(names,key=lambda n:teachers[n][u]['ent'])][u]["pred"])
# domain route (subject's best-overall teacher) — ORACLE (uses labels to pick per-subject best)
res["domain_route_ORACLE"]=acc_of(lambda u: teachers[subj_best[teachers[names[0]][u]["subj"]]][u]["pred"])
# domain route CV (HONEST/achievable): estimate subject-best teacher on train folds, apply to test fold
def domain_route_cv(k=5, seed=42):
    rng=np.random.RandomState(seed)
    idx=np.array(common); rng.shuffle(idx)
    folds=np.array_split(idx,k)
    correct=0
    for i in range(k):
        test=set(folds[i].tolist()); train=[u for u in common if u not in test]
        # subject-best teacher estimated on TRAIN only
        sb={}
        s_items=defaultdict(list)
        for u in train: s_items[teachers[names[0]][u]["subj"]].append(u)
        for s,us in s_items.items():
            accs={n:np.mean([teachers[n][u]["gt"]==teachers[n][u]["pred"] for u in us]) for n in names}
            sb[s]=max(accs,key=accs.get)
        for u in folds[i]:
            subj=teachers[names[0]][u]["subj"]
            router=sb.get(subj,best_single)  # unseen subject -> best overall
            if teachers[router][u]["pred"]==teachers[names[0]][u]["gt"]: correct+=1
    return 100*correct/N
res["domain_route_CV"]=domain_route_cv()
# prob average
def probavg(u):
    P=np.mean([teachers[n][u]["p"] for n in names],axis=0)
    return LETTERS[int(np.argmax(P))]
res["prob_avg"]=acc_of(probavg)

bs=overall[best_single]
def delta(x): return round(res[x]-bs,2)
# ACHIEVABLE ceiling = best label-free / CV-honest fusion (NOT the GT-using oracles)
achievable=max(res["majority_vote"],res["conf_route"],res["domain_route_CV"],res["prob_avg"])
gap=round(achievable-bs,2)
# oracle ceiling shows the headroom that a BETTER router could capture
oracle_ceiling=round(res["domain_route_ORACLE"]-bs,2)
verdict = "GO" if gap>=2.0 else ("NO-GO" if gap<0.5 else "WEAK-GO")

out={"n_common":N,"teachers":names,"overall_acc":{k:round(v,2) for k,v in overall.items()},
     "best_single":best_single,"best_single_acc":round(bs,2),
     "fusion":{k:round(v,2) for k,v in res.items()},
     "delta_vs_best_single":{k:delta(k) for k in res},
     "achievable_ceiling":round(achievable,2),"gap_pp":gap,
     "oracle_ceiling_pp":oracle_ceiling,"verdict":verdict,
     "subject_best_teacher":subj_best}
json.dump(out,open(f"{REP}/fusion_oracle.json","w"),ensure_ascii=False,indent=2)

print(f"=== FUSION ORACLE (n={N} common items, {len(names)} teachers) ===")
print(f"best single teacher: {best_single} = {bs:.2f}%")
for k in ["oracle_anyright","majority_vote","conf_route","prob_avg","domain_route_ORACLE","domain_route_CV"]:
    print(f"  {k:20s} {res[k]:.2f}%  ({delta(k):+.2f}pp)")
print(f"ACHIEVABLE ceiling (label-free/CV, no GT): {achievable:.2f}%  gap={gap:+.2f}pp")
print(f"ORACLE ceiling (domain_route w/ GT):       headroom {oracle_ceiling:+.2f}pp")
print(f"VERDICT: {verdict}  (>=2.0 GO / <0.5 NO-GO)")
