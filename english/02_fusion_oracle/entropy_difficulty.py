#!/usr/bin/env python3
"""Task 1 — Entropy-difficulty external validation (moat for path 3b), ZERO GPU.
English dental has NO human difficulty labels -> use CROSS-MODEL CONSENSUS as the
difficulty gold standard (how many of the 7 teachers get an item wrong).

Claims tested (mirrors CMExam H4/5c/5d):
  H4  within a strong teacher, high-entropy (low-cred) subset has much higher error rate.
  5d  teacher entropy correlates with cross-model consensus difficulty (external gold).
  5d-null  entropy does NOT reduce to surface text artifacts (stem length, #negation words).
Outputs english/02_fusion_oracle/entropy_difficulty.{json,md}
"""
import json, os, glob, re, math
import numpy as np
from collections import defaultdict

LP="english/01_teacher_screening/logprobs"
REP="english/02_fusion_oracle"; os.makedirs(REP,exist_ok=True)
LETTERS=["A","B","C","D","E"]

def spearman(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    ar=np.argsort(np.argsort(a)); br=np.argsort(np.argsort(b))
    if ar.std()==0 or br.std()==0: return 0.0
    return float(np.corrcoef(ar,br)[0,1])

def perm_pvalue(a,b,rho,iters=2000,seed=0):
    rng=np.random.RandomState(seed); b=np.asarray(b,float); cnt=0
    for _ in range(iters):
        if abs(spearman(a,rng.permutation(b)))>=abs(rho): cnt+=1
    return (cnt+1)/(iters+1)

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
        pred=LETTERS[int(np.argmax(raw))]
        out[uid]={"gt":gt,"ent":ent,"pred":pred,"correct":int(pred==gt),
                  "subj":r.get("Medical Discipline","?"),
                  "stem":r.get("Question",""),"opts":r.get("Options",{})}
    return out

teachers={}
for f in sorted(glob.glob(f"{LP}/*_logprobs.jsonl")):
    name=os.path.basename(f).replace("_logprobs.jsonl","")
    d=load(f)
    if d: teachers[name]=d
names=list(teachers)
common=sorted(set.intersection(*[set(d) for d in teachers.values()]))
N=len(common)

# cross-model consensus difficulty = # teachers WRONG on item (0..7)
consensus_wrong={u:sum(1-teachers[n][u]["correct"] for n in names) for u in common}

report={"n_common":N,"n_teachers":len(names),"teachers":names}

# ---- H4: per-teacher low-cred (high-entropy) vs high-cred error rate ----
h4={}
for n in names:
    ents=np.array([teachers[n][u]["ent"] for u in common])
    errs=np.array([1-teachers[n][u]["correct"] for u in common])
    thr=np.quantile(ents,0.5)  # bottom-50% cred = high entropy
    hi=ents>thr; lo=ents<=thr
    hi_err=100*errs[hi].mean() if hi.sum() else None
    lo_err=100*errs[lo].mean() if lo.sum() else None
    ratio=(hi_err/lo_err) if (lo_err and lo_err>0) else None
    h4[n]={"acc":round(100*(1-errs.mean()),2),
           "high_entropy_err":round(hi_err,2) if hi_err is not None else None,
           "low_entropy_err":round(lo_err,2) if lo_err is not None else None,
           "err_ratio":round(ratio,2) if ratio else None}
report["H4_entropy_locates_errors"]=h4

# ---- 5d: entropy vs cross-model consensus difficulty ----
five_d={}
cons=[consensus_wrong[u] for u in common]
for n in names:
    ents=[teachers[n][u]["ent"] for u in common]
    rho=spearman(ents,cons)
    five_d[n]={"entropy_vs_consensus_rho":round(rho,4),
               "p_perm":round(perm_pvalue(ents,cons,rho,iters=1000,seed=1),5)}
# mean entropy across teachers vs consensus (aggregate signal)
mean_ent=[np.mean([teachers[n][u]["ent"] for n in names]) for u in common]
rho_agg=spearman(mean_ent,cons)
report["5d_entropy_vs_consensus"]={"per_teacher":five_d,
    "mean_entropy_vs_consensus_rho":round(rho_agg,4),
    "p_perm":round(perm_pvalue(mean_ent,cons,rho_agg,iters=1000,seed=2),5)}

# consensus difficulty gradient: mean entropy at each consensus level
grad={}
for w in range(len(names)+1):
    us=[u for u in common if consensus_wrong[u]==w]
    if us: grad[w]={"n":len(us),"mean_teacher_ent":round(float(np.mean([mean_ent[common.index(u)] for u in us])),4)}
report["consensus_gradient"]=grad

# ---- 5d-null: surface-text artifacts ----
NEG=re.compile(r'\b(not|except|least|never|cannot|false|incorrect|unlikely|contraindicated)\b',re.I)
def stem_len(u): 
    s=teachers[names[0]][u]["stem"]; return len(s.split())
def neg_count(u):
    s=teachers[names[0]][u]["stem"]; return len(NEG.findall(s))
slen=[stem_len(u) for u in common]; negc=[neg_count(u) for u in common]
report["5d_null_surface"]={
    "mean_entropy_vs_stem_length_rho":round(spearman(mean_ent,slen),4),
    "mean_entropy_vs_negation_count_rho":round(spearman(mean_ent,negc),4),
    "consensus_vs_stem_length_rho":round(spearman(cons,slen),4),
    "note":"near-zero => entropy/difficulty is NOT a surface-text artifact"}

json.dump(report,open(f"{REP}/entropy_difficulty.json","w"),ensure_ascii=False,indent=2)

# markdown
md=[f"# English Dental — Entropy=Difficulty External Validation (n={N}, {len(names)} teachers)\n",
    "Gold standard = CROSS-MODEL CONSENSUS (# of 7 teachers wrong), since English has no human difficulty labels.\n",
    "## H4: entropy locates a teacher's own error subset",
    "| teacher | acc% | high-entropy err% | low-entropy err% | ratio |","|---|---|---|---|---|"]
for n in sorted(h4,key=lambda k:-h4[k]["acc"]):
    v=h4[n]; md.append(f"| {n} | {v['acc']} | {v['high_entropy_err']} | {v['low_entropy_err']} | {v['err_ratio']}× |")
md.append("\n## 5d: entropy vs cross-model consensus difficulty (external gold)")
md.append(f"- **mean-entropy vs consensus ρ = {report['5d_entropy_vs_consensus']['mean_entropy_vs_consensus_rho']}** (p_perm={report['5d_entropy_vs_consensus']['p_perm']})")
md.append("\n| teacher | entropy vs consensus ρ | p_perm |")
md.append("|---|---|---|")
for n in names:
    v=five_d[n]; md.append(f"| {n} | {v['entropy_vs_consensus_rho']} | {v['p_perm']} |")
md.append("\n### Consensus difficulty gradient (mean teacher entropy rises with #wrong)")
md.append("| #teachers wrong | n items | mean teacher entropy |")
md.append("|---|---|---|")
for w,v in grad.items(): md.append(f"| {w} | {v['n']} | {v['mean_teacher_ent']} |")
md.append("\n## 5d-null: surface-text artifact controls (want ≈0)")
s=report["5d_null_surface"]
md.append(f"- entropy vs stem length ρ = {s['mean_entropy_vs_stem_length_rho']}")
md.append(f"- entropy vs #negation words ρ = {s['mean_entropy_vs_negation_count_rho']}")
md.append(f"- consensus vs stem length ρ = {s['consensus_vs_stem_length_rho']}")
open(f"{REP}/entropy_difficulty.md","w").write("\n".join(md))

print(f"=== ENTROPY=DIFFICULTY (n={N}, {len(names)} teachers) ===")
print("H4 (high vs low entropy err ratio):")
for n in sorted(h4,key=lambda k:-h4[k]["acc"]):
    print(f"  {n:10s} acc={h4[n]['acc']}%  hi-ent-err={h4[n]['high_entropy_err']}% vs lo={h4[n]['low_entropy_err']}%  ({h4[n]['err_ratio']}×)")
print(f"5d mean-entropy vs consensus difficulty: ρ={report['5d_entropy_vs_consensus']['mean_entropy_vs_consensus_rho']} p={report['5d_entropy_vs_consensus']['p_perm']}")
print(f"5d-null surface: len ρ={s['mean_entropy_vs_stem_length_rho']}  neg ρ={s['mean_entropy_vs_negation_count_rho']}")
print(f"-> {REP}/entropy_difficulty.md")
