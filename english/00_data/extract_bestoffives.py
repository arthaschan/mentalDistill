#!/usr/bin/env python3
"""Extract Best of Fives for Dentistry -> structured MCQ JSONL (RAW reading-order mode).
Questions: 'N.N <stem...>' then option lines 'A ...'..'E ...'.
Answers  : 'N.N <letter>' alone, then explanation prose.
OCR fixes: option letter 'B' often read as '8' or '.8'; letters may be lowercase.
"""
import re, json, subprocess
from collections import Counter

PDF = "english/Best of Fives for Dentistry.pdf"
CHAPTERS = {
    1: "Oral Medicine", 2: "Oral Surgery", 3: "Dental Materials",
    4: "Child Dental Health and Orthodontics", 5: "Oral Pathology",
    6: "Periodontics", 7: "Pharmacology", 8: "Radiology",
    9: "Restorative Dentistry", 10: "Human Disease",
}
txt = subprocess.run(["pdftotext","-raw",PDF,"-"],capture_output=True,text=True).stdout
lines = [l.rstrip() for l in txt.split("\n")]
N = len(lines)

qid_re = re.compile(r'^\s*(\d{1,2})\.(\d{1,3})\s+(.*)$')
# option line: leading letter A-E (allow OCR: 8/6/o for B/G/... minimal) then space then text
def opt_letter(s):
    m = re.match(r'^\s*([A-Ea-e])\s+\S', s)
    if m: return m.group(1).upper()
    # OCR: 'B' misread as '8' or '.8'
    m = re.match(r'^\s*\.?8\s+\S', s)
    if m: return 'B'
    return None

# answer-line: 'N.N <letter/8>' with nothing meaningful after
ans_re = re.compile(r'^\s*(\d{1,2})\.(\d{1,3})\s+([A-Ea-e8])\s*$')

questions, answers = {}, {}
i = 0
while i < N:
    line = lines[i]
    ma = ans_re.match(line)
    if ma:
        ch,q,let = int(ma.group(1)),int(ma.group(2)),ma.group(3).upper()
        if let=='8': let='B'
        if ch in CHAPTERS:
            answers[(ch,q)] = let
        i += 1; continue
    mq = qid_re.match(line)
    if mq:
        ch,q,rest = int(mq.group(1)),int(mq.group(2)),mq.group(3).strip()
        if ch not in CHAPTERS or not rest:
            i += 1; continue
        stem=[rest]; opts={}; j=i+1
        while j < N and len(opts) < 5:
            lj = lines[j]
            L = opt_letter(lj)
            if L:
                val = re.sub(r'^\s*(\.?8|[A-Ea-e])\s+','',lj).strip()
                opts[L]=val; j+=1
                continue
            if opts:  # options started, non-option => end
                break
            if lj.strip() and not qid_re.match(lj) and 'ymail' not in lj and 'BEST OF FIVES' not in lj:
                stem.append(lj.strip()); j+=1
            else:
                break
        if len(opts) >= 4:
            questions[(ch,q)] = {"stem":" ".join(stem).strip(),"options":opts}
            i=j; continue
    i += 1

records=[]
for key,qd in sorted(questions.items()):
    ch,q=key; ans=answers.get(key)
    records.append({"id":f"BoF-{ch}.{q}","subject":CHAPTERS[ch],
                    "stem":qd["stem"],"options":qd["options"],
                    "answer":ans,"n_options":len(qd["options"])})

have_ans=sum(1 for r in records if r["answer"] and r["answer"] in r["options"])
subj=Counter(r["subject"] for r in records)
ansc=Counter(r["answer"] for r in records if r["answer"])
nopt=Counter(r["n_options"] for r in records)
with open("english/bestoffives.jsonl","w") as f:
    for r in records: f.write(json.dumps(r,ensure_ascii=False)+"\n")
print(f"total stems parsed : {len(records)}")
print(f"with valid answer  : {have_ans} ({100*have_ans/max(len(records),1):.0f}%)")
print(f"n_options dist     : {dict(sorted(nopt.items()))}")
print(f"answer letter dist : {dict(sorted(ansc.items()))}")
print("subject dist:")
for s,c in subj.most_common(): print(f"   {s:42s} {c}")
