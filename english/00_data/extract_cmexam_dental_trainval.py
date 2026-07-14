#!/usr/bin/env python3
"""Paper Line B: extract Chinese dental MCQs from CMExam train.csv / val.csv.
These CSVs LACK the 'Medical Discipline' column, so we keyword-filter (Chinese dental terms).
Dedup against the labeled test dental set (口腔医学) to prevent leakage.
Output: english/00_data/cmexam_dental_trainval.jsonl  (for the Chinese dental-specialist student)
"""
import pandas as pd, re, json, hashlib, os
CSVDIR="english/dataset"
OUT="english/00_data/cmexam_dental_trainval.jsonl"
KW=re.compile(r'(牙|口腔|龋|齿|根管|牙周|牙髓|正畸|义齿|种植体|颌|唾液|涎|黏膜|磨牙|切牙|尖牙|乳牙|恒牙|釉质|牙本质|牙龈|拔牙|咬合)')

# leakage guard: labeled test dental questions
te=pd.read_csv(f"{CSVDIR}/test_with_annotations.csv")
test_dent_q=set(te[te["Medical Discipline"].astype(str)=="口腔医学"]["Question"].astype(str))
all_test_q=set(te["Question"].astype(str))

def norm(s): return re.sub(r'\s+',' ',str(s)).strip()
seen=set(); recs=[]; leak=0
for split in ["train","val"]:
    df=pd.read_csv(f"{CSVDIR}/{split}.csv")
    for _,r in df.iterrows():
        q=norm(r["Question"])
        if not KW.search(q): continue
        if q in all_test_q: leak+=1; continue   # drop any test overlap
        key=q.lower()
        if key in seen: continue
        seen.add(key)
        recs.append({"uid":"CMExamDent-"+hashlib.md5(key.encode()).hexdigest()[:10],
                     "source":f"CMExam_{split}","Question":q,
                     "Options":norm(r["Options"]),"Answer":norm(r["Answer"]).upper(),
                     "Explanation":norm(r.get("Explanation",""))[:600]})
os.makedirs(os.path.dirname(OUT),exist_ok=True)
with open(OUT,"w") as f:
    for r in recs: f.write(json.dumps(r,ensure_ascii=False)+"\n")
from collections import Counter
print(f"CMExam dental (train+val, keyword, dedup): {len(recs)}")
print(f"dropped test-overlap (leakage guard): {leak}")
print(f"source: {dict(Counter(r['source'] for r in recs))}")
print(f"answer dist: {dict(sorted(Counter(r['Answer'] for r in recs).items()))}")
print(f"-> {OUT}")
