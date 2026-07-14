#!/usr/bin/env python3
"""Merge the 3 English dental sources into a unified dataset + stratified splits.

Sources (already extracted to JSONL):
  bestoffives.jsonl  - single-best A-E  (source=BoF)
  nbde.jsonl         - single-best A-E  (source=NBDE)
  mcqs_tf.jsonl      - true/false multi-select (source=MCQ, auxiliary task)

Output (english/dataset/):
  single_best_all.jsonl   canonical single-best pool (BoF+NBDE), cleaned+deduped
  train.jsonl val.jsonl test.jsonl   stratified-by-subject split of single_best pool
  tf_all.jsonl            cleaned true/false auxiliary set
  STATS.md                audit report
Split: 70/15/15 stratified by subject, seed=42.
"""
import json, re, os, random, hashlib
from collections import Counter, defaultdict

ROOT="english"
OUT=os.path.join(ROOT,"dataset")
os.makedirs(OUT,exist_ok=True)
SEED=42; random.seed(SEED)

def clean(s):
    if not s: return s
    s=s.replace("\x0c"," ")
    s=re.sub(r'(\w)-\s+(\w)', r'\1\2', s)      # de-hyphenate line-break splits
    s=re.sub(r'\s+',' ',s).strip()
    # strip a trailing stray next-question id like "... 6.7" at very end of stem
    s=re.sub(r'\s+\d{1,2}\.\d{1,3}\s*$','',s).strip()
    return s

def load(fn):
    p=os.path.join(ROOT,fn)
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []

def norm_rec(r, source):
    opts={k:clean(v) for k,v in r["options"].items()}
    rec={"uid":None,"source":source,"subject":r["subject"],
         "stem":clean(r["stem"]),"options":opts,"answer":r["answer"],
         "n_options":len(opts),"format":r.get("format","single_best")}
    key=(rec["stem"]+"||"+"|".join(f"{k}:{v}" for k,v in sorted(opts.items()))).lower()
    rec["uid"]=source+"-"+hashlib.md5(key.encode()).hexdigest()[:10]
    return rec, key

def valid_single(r):
    return (r["answer"] and r["answer"] in r["options"]
            and r["n_options"]>=4 and len(r["stem"])>10
            and all(len(v)>=1 for v in r["options"].values()))

def valid_tf(r):
    return (r["answer"] and all(c in r["options"] for c in r["answer"])
            and r["n_options"]>=4 and len(r["stem"])>5)

# ---- single-best pool ----
seen=set(); single=[]
for fn,src in [("bestoffives.jsonl","BoF"),("nbde.jsonl","NBDE")]:
    for r in load(fn):
        rec,key=norm_rec(r,src)
        if not valid_single(rec): continue
        if key in seen: continue
        seen.add(key); single.append(rec)

# ---- true/false auxiliary ----
tf=[]; seen_tf=set()
for r in load("mcqs_tf.jsonl"):
    rec,key=norm_rec(r,"MCQ")
    rec["format"]="true_false_multi"
    if not valid_tf(rec): continue
    if key in seen_tf: continue
    seen_tf.add(key); tf.append(rec)

# ---- stratified split of single pool by subject ----
bysub=defaultdict(list)
for r in single: bysub[r["subject"]].append(r)
train,val,test=[],[],[]
for sub,items in bysub.items():
    random.shuffle(items)
    n=len(items); n_tr=int(round(n*0.70)); n_va=int(round(n*0.15))
    train+=items[:n_tr]; val+=items[n_tr:n_tr+n_va]; test+=items[n_tr+n_va:]
random.shuffle(train); random.shuffle(val); random.shuffle(test)

def dump(fn,rows):
    with open(os.path.join(OUT,fn),"w") as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False)+"\n")

dump("single_best_all.jsonl",single)
dump("train.jsonl",train); dump("val.jsonl",val); dump("test.jsonl",test)
dump("tf_all.jsonl",tf)

# ---- audit report ----
def sub_table(rows):
    c=Counter(r["subject"] for r in rows)
    return "\n".join(f"| {s} | {n} |" for s,n in c.most_common())
def ans_dist(rows):
    return dict(sorted(Counter(r["answer"] for r in rows).items()))
src_cnt=Counter(r["source"] for r in single)

md=f"""# English Dental MCQ Dataset — Build Stats (seed={SEED})

## Single-best pool (main KD task: choose one of A-E)
- total after clean+dedup: **{len(single)}**  (BoF {src_cnt['BoF']} + NBDE {src_cnt['NBDE']})
- split: train **{len(train)}** / val **{len(val)}** / test **{len(test)}**  (70/15/15 stratified by subject)
- answer-letter dist (all): {ans_dist(single)}
- n_options dist: {dict(sorted(Counter(r['n_options'] for r in single).items()))}

### Subjects (single-best pool)
| subject | n |
|---|---|
{sub_table(single)}

### Split subject balance
- train subjects: {dict(Counter(r['subject'] for r in train).most_common())}
- val   subjects: {dict(Counter(r['subject'] for r in val).most_common())}
- test  subjects: {dict(Counter(r['subject'] for r in test).most_common())}

## True/False auxiliary set (source: MCQs for Dentistry)
- total after clean+dedup: **{len(tf)}**  (multi-select; answer = set of TRUE statements)
- kept as generalization/robustness set, NOT in main train/val/test.
- subjects:
{sub_table(tf)}

## Files
- dataset/single_best_all.jsonl, train/val/test.jsonl (main)
- dataset/tf_all.jsonl (auxiliary)

## Notes / honest boundaries
- Small training set (~{int(len(single)*0.7)} items): use few-shot teacher labeling + multi-seed + report CI.
- OCR residue possible (spelling of Latin/drug terms); answers/keys verified structurally.
- Sources are copyrighted revision books — labels/derived data for internal research only; do not redistribute raw text.
"""
open(os.path.join(OUT,"STATS.md"),"w").write(md)
print(f"single-best pool: {len(single)} (BoF {src_cnt['BoF']} + NBDE {src_cnt['NBDE']})")
print(f"split: train {len(train)} / val {len(val)} / test {len(test)}")
print(f"true/false aux: {len(tf)}")
print(f"written -> {OUT}/  (see STATS.md)")
