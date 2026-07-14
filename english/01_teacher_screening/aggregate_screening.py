#!/usr/bin/env python3
"""Aggregate per-teacher logprobs -> teacher prior table + per-subject accuracy matrix.
This matrix drives the complementarity check (Screening #2) and the fusion oracle (Screening #3).
Outputs english/01_teacher_screening/reports/{teacher_prior.md, subject_matrix.csv, screening.json}
"""
import json, os, glob
import numpy as np
from collections import defaultdict

LP="english/01_teacher_screening/logprobs"
REP="english/01_teacher_screening/reports"; os.makedirs(REP,exist_ok=True)
LETTERS=["A","B","C","D","E"]

def load(path):
    rows={}
    for line in open(path):
        line=line.strip()
        if not line: continue
        try: r=json.loads(line)
        except: continue
        dist=r.get("TeacherDist",{})
        # generator OVERWRITES 'Answer' with its own prediction; TRUE gold is in 'OriginalAnswer'
        gt=str(r.get("OriginalAnswer") or r.get("Answer","")).strip().upper()
        uid=r.get("uid") or r.get("Question","")[:40]
        if not dist or gt not in LETTERS: continue
        raw=np.array([float(dist.get(c,0.0)) for c in LETTERS])
        if raw.sum()<=1e-9: continue
        raw=raw/raw.sum()
        pred=LETTERS[int(np.argmax(raw))]
        ent=float(-np.sum(np.clip(raw,1e-12,None)*np.log(np.clip(raw,1e-12,None))))
        rows[uid]={"gt":gt,"pred":pred,"correct":int(pred==gt),"ent":ent,
                   "subj":r.get("Medical Discipline","?"),"dist":raw.tolist()}
    return rows

files=sorted(glob.glob(f"{LP}/*_logprobs.jsonl"))
teachers={}
for f in files:
    name=os.path.basename(f).replace("_logprobs.jsonl","")
    r=load(f)
    if r: teachers[name]=r
if not teachers:
    print("[no logprobs yet]"); raise SystemExit

# overall prior
prior={}
for name,rows in teachers.items():
    acc=100*np.mean([v["correct"] for v in rows.values()])
    ent=float(np.mean([v["ent"] for v in rows.values()]))
    prior[name]={"n":len(rows),"acc":round(acc,2),"mean_ent":round(ent,4)}
order=sorted(prior,key=lambda k:-prior[k]["acc"])

# per-subject accuracy matrix
subjects=sorted({v["subj"] for rows in teachers.values() for v in rows.values()})
matrix=defaultdict(dict)
for name,rows in teachers.items():
    bysub=defaultdict(list)
    for v in rows.values(): bysub[v["subj"]].append(v["correct"])
    for s in subjects:
        matrix[name][s]=round(100*np.mean(bysub[s]),1) if bysub[s] else None

# complementarity check: is there a single teacher that dominates EVERY subject?
best_teacher=order[0]
per_subject_winner={}
for s in subjects:
    vals={n:matrix[n].get(s) for n in teachers if matrix[n].get(s) is not None}
    if vals:
        w=max(vals,key=vals.get); per_subject_winner[s]=(w,vals[w])
dom = all(w==best_teacher for w,_ in per_subject_winner.values())
n_winners=len(set(w for w,_ in per_subject_winner.values()))

# write outputs
out={"teacher_prior":prior,"order":order,"subjects":subjects,
     "subject_matrix":{n:matrix[n] for n in teachers},
     "per_subject_winner":per_subject_winner,
     "single_teacher_dominates":dom,"distinct_subject_winners":n_winners}
json.dump(out,open(f"{REP}/screening.json","w"),ensure_ascii=False,indent=2)

# csv
with open(f"{REP}/subject_matrix.csv","w") as w:
    w.write("subject,"+",".join(order)+"\n")
    for s in subjects:
        w.write(s+","+",".join(str(matrix[n].get(s,"")) for n in order)+"\n")
    w.write("OVERALL,"+",".join(str(prior[n]["acc"]) for n in order)+"\n")

# markdown prior table
md=["# English Dental — Teacher Screening\n",
    f"Pool: {prior[order[0]]['n']} single-best items. Teachers screened: {len(teachers)}.\n",
    "## Teacher prior (zero-shot, English prompt)\n",
    "| rank | teacher | acc% | mean_entropy |","|---|---|---|---|"]
for i,n in enumerate(order,1):
    md.append(f"| {i} | {n} | {prior[n]['acc']} | {prior[n]['mean_ent']} |")
md.append("\n## Complementarity check (GO/NO-GO precursor)\n")
md.append(f"- Best overall teacher: **{best_teacher}** ({prior[best_teacher]['acc']}%)")
md.append(f"- Distinct per-subject winners: **{n_winners}** / {len(subjects)} subjects")
md.append(f"- Single teacher dominates every subject: **{dom}**")
if dom:
    md.append(f"- => Mirrors Chinese CMExam (one boss). Fusion likely DEAD; verify with oracle then lean 3b.")
else:
    md.append(f"- => No single boss on English dental. Fusion has a chance; run fusion oracle (Screening #3).")
md.append("\n### Per-subject winners")
md.append("| subject | winner | acc% |")
md.append("|---|---|---|")
for s,(w,a) in sorted(per_subject_winner.items()):
    md.append(f"| {s} | {w} | {a} |")
open(f"{REP}/teacher_prior.md","w").write("\n".join(md))

print("=== TEACHER PRIOR (English dental) ===")
for i,n in enumerate(order,1):
    print(f"  {i}. {n:10s} {prior[n]['acc']:.2f}%  ent={prior[n]['mean_ent']:.3f}")
print(f"distinct subject winners: {n_winners}/{len(subjects)}  single_boss={dom}")
print(f"-> reports/teacher_prior.md, subject_matrix.csv, screening.json")
