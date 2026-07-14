#!/usr/bin/env python3
"""Extract a CLEAN dental single-best subset from MedMCQA (openlifescienceai/medmcqa).
CAUTION: MedMCQA 'subject_name==Dental' is POLLUTED (only ~37% actually dental; many are
dermatology/STD mislabeled). So we require BOTH subject==Dental AND a dental keyword hit,
AND choice_type=='single', AND valid answer + non-empty 4 options. Output tagged source=MedMCQA.
"""
import json, os, re, hashlib
from datasets import load_dataset

OUT="english/00_data/medmcqa_dental.jsonl"
LETTERS=["A","B","C","D"]
DENT=re.compile(r'\b(tooth|teeth|dental|dentine|dentin|enamel|pulp|molar|premolar|incisor|canine|gingiv|periodont|oral|mandib|maxill|caries|occlus|denture|endodont|orthodont|amalgam|prosthodont|alveolar|cementum|odonto|crown|root canal|fluoride|saliva|palat|buccal|lingual|mucosa|periapical|dentition|bruxism|malocclusion)\b',re.I)

def clean(s): return re.sub(r'\s+',' ',(s or '')).strip()

# train + validation (both carry public labels); test split has no public cop -> excluded
dental=[]
for sp in ["train","validation"]:
    ds=load_dataset("openlifescienceai/medmcqa",split=sp)
    dental += [dict(r, _split=sp) for r in ds if r["subject_name"]=="Dental"]

seen=set(); recs=[]
for r in dental:
    if r["choice_type"]!="single": continue
    opts={"A":clean(r["opa"]),"B":clean(r["opb"]),"C":clean(r["opc"]),"D":clean(r["opd"])}
    stem=clean(r["question"])
    txt=" ".join([stem]+list(opts.values()))
    if not DENT.search(txt): continue
    if len(stem)<10 or any(len(v)<1 for v in opts.values()): continue
    cop=r["cop"]
    if cop is None or cop<0 or cop>3: continue
    ans=LETTERS[cop]
    key=(stem+"||"+"|".join(f"{k}:{v}" for k,v in opts.items())).lower()
    if key in seen: continue
    seen.add(key)
    uid="MedMCQA-"+hashlib.md5(key.encode()).hexdigest()[:10]
    recs.append({"uid":uid,"source":"MedMCQA","subject":"Dental (MedMCQA/NEET-MDS)",
                 "stem":stem,"options":opts,"answer":ans,"n_options":4,
                 "format":"single_best","exp":clean(r["exp"])[:500]})

os.makedirs(os.path.dirname(OUT),exist_ok=True)
with open(OUT,"w") as f:
    for r in recs: f.write(json.dumps(r,ensure_ascii=False)+"\n")

from collections import Counter
print(f"clean dental single-best extracted: {len(recs)}")
print("answer dist:",dict(sorted(Counter(r['answer'] for r in recs).items())))
print(f"with explanation: {sum(1 for r in recs if r['exp'])}")
print(f"-> {OUT}")
