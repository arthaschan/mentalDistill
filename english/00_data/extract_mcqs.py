#!/usr/bin/env python3
"""Extract 'MCQs for Dentistry' -> true/false multi-select JSONL.
Each question: stem + statements A-E; answer = set of TRUE statements (e.g. 'ADE').
Questions and their answer keys interleave within each chapter, shared N.N id.
"""
import re, json, subprocess
from collections import Counter

PDF="english/MCQs for Dentistry.pdf"
txt=subprocess.run(["pdftotext","-raw",PDF,"-"],capture_output=True,text=True).stdout
lines=txt.split("\n"); N=len(lines)

CHAPS={"General Dentistry","Human Disease","Oral Medicine","Oral Pathology","Oral Surgery",
"Child Dental Health and Orthodontics","Therapeutics","Dental Materials",
"Radiology and Radiography","Restorative Dentistry"}

qid_re=re.compile(r'^\s*(\d{1,2})\.(\d{1,3})\s*(.*)$')   # allow empty rest (stem on next line)
opt_re=re.compile(r'^\s*([A-E])\s+(\S.*)$')     # inline: "A text"
optbare_re=re.compile(r'^\s*([A-E])\s*$')        # bare letter on its own line
ans_re=re.compile(r'^\s*(\d{1,2})\.(\d{1,3})\s+([A-E]{1,5})\s*$')

questions={}; answers={}; chap=None
i=0
while i<N:
    s=lines[i].strip()
    if s in CHAPS: chap=s; i+=1; continue
    ma=ans_re.match(lines[i])
    if ma:
        ch,q=int(ma.group(1)),int(ma.group(2))
        # answer combos are letters in order; validate strictly ascending unique A-E
        combo=ma.group(3)
        if combo==''.join(sorted(set(combo))) and all(c in "ABCDE" for c in combo):
            answers[(ch,q)]=combo; i+=1; continue
    mq=qid_re.match(lines[i])
    if mq:
        ch,q,rest=int(mq.group(1)),int(mq.group(2)),mq.group(3).strip()
        stem=[rest]; opts={}; j=i+1
        # expected next option letter (A,B,C,D,E in order) to disambiguate bare letters
        seq="ABCDE"
        while j<N and len(opts)<5:
            lj=lines[j]
            mo=opt_re.match(lj)
            mb=optbare_re.match(lj)
            nextL=seq[len(opts)] if len(opts)<5 else None
            if mo and mo.group(1)==nextL:
                L=mo.group(1); val=[mo.group(2).strip()]; k=j+1
            elif mb and mb.group(1)==nextL:
                L=mb.group(1); val=[]; k=j+1
            else:
                if opts: break
                if lj.strip() and not qid_re.match(lj) and lj.strip() not in CHAPS:
                    stem.append(lj.strip()); j+=1; continue
                else: break
            while k<N:
                lk=lines[k]
                nx=seq[len(opts)+1] if len(opts)+1<5 else None
                if (opt_re.match(lk) and opt_re.match(lk).group(1)==nx) or \
                   (optbare_re.match(lk) and optbare_re.match(lk).group(1)==nx) or \
                   qid_re.match(lk) or lk.strip() in CHAPS: break
                if lk.strip(): val.append(lk.strip())
                else:
                    if val: break
                k+=1
            opts[L]=" ".join(val).strip(); j=k
        if len(opts)>=4:
            questions[(ch,q)]={"stem":" ".join(stem).strip(),"options":opts,"chap":chap}
            i=j; continue
    i+=1

records=[]
for key,qd in sorted(questions.items()):
    ch,q=key; ans=answers.get(key)
    valid = ans is not None and all(c in qd["options"] for c in ans)
    records.append({"id":f"MCQ-{ch}.{q}","subject":qd["chap"] or "Unknown",
                    "stem":qd["stem"],"options":qd["options"],
                    "answer":ans,"n_options":len(qd["options"]),"format":"true_false_multi"})

have=sum(1 for r in records if r["answer"] and all(c in r["options"] for c in r["answer"]))
with open("english/mcqs_tf.jsonl","w") as f:
    for r in records: f.write(json.dumps(r,ensure_ascii=False)+"\n")
print(f"questions parsed : {len(questions)}   answers parsed: {len(answers)}")
print(f"records: {len(records)}  with valid answer: {have} ({100*have/max(len(records),1):.0f}%)")
print("n_options:",dict(sorted(Counter(r['n_options'] for r in records).items())))
print("subjects:")
for s,c in Counter(r['subject'] for r in records).most_common(): print(f"   {s:40s} {c}")
