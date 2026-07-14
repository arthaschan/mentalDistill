#!/usr/bin/env python3
"""Extract Mosby's NBDE Part Two 'Sample Examination' -> single-best MCQ JSONL.
Questions block and Answer-Key block are each grouped by discipline; question
numbers reset per discipline. Pair by (discipline, number).
Question: 'N. stem...' then 'A. ...'..'E. ...'
Answer  : 'N. LETTER. explanation...'
"""
import re, json, subprocess
from collections import Counter

PDF = "english/Mosby’s Review for the NBDE Part Two ( PDFDrive ).pdf"
txt = subprocess.run(["pdftotext","-raw",PDF,"-"],capture_output=True,text=True).stdout
lines = txt.split("\n")
N=len(lines)

DISC = {"Endodontics","Operative Dentistry","Oral and Maxillofacial Surgery",
        "Oral Diagnosis","Patient Management","Periodontics","Pharmacology","Prosthodontics"}

# locate the Sample Examination question block and the Answer Key block
q_start = next(i for i,l in enumerate(lines) if l.strip()=="Sample Examination" and i>30000)
a_start = next(i for i,l in enumerate(lines) if l.strip().startswith("Answer Key for Sample") and i>40000)
# answer key title spans 2 lines ("Answer Key for Sample" / "Examination")

num_re  = re.compile(r'^\s*(\d{1,3})\.\s+(.*)$')
qopt_re = re.compile(r'^\s*([A-E])\.\s+(.*)$')
ans_re  = re.compile(r'^\s*(\d{1,3})\.\s+([A-E])\.\s+(.*)$')
CLEAN = re.compile(r'^\s*\f?\d{0,4}\s*Sample Examination\s*$|^\s*Sample Examination\s*\d*\s*$')

def parse_questions(a, b):
    out={}  # (disc,num)->{stem,options}
    disc=None; i=a
    while i < b:
        s=lines[i].strip()
        if s in DISC:
            disc=s; i+=1; continue
        if CLEAN.match(lines[i]) or s=="Sample Examination":
            i+=1; continue
        m=num_re.match(lines[i])
        if m and disc:
            num=int(m.group(1)); stem=[m.group(2).strip()]; opts={}; j=i+1
            while j<b and len(opts)<5:
                lj=lines[j]
                if CLEAN.match(lj): j+=1; continue
                mo=qopt_re.match(lj)
                if mo:
                    L=mo.group(1); val=[mo.group(2).strip()]
                    # option continuation lines (until next option/num/discipline)
                    k=j+1
                    while k<b:
                        lk=lines[k]
                        if qopt_re.match(lk) or num_re.match(lk) or lk.strip() in DISC or CLEAN.match(lk):
                            break
                        if lk.strip(): val.append(lk.strip())
                        else: break
                        k+=1
                    opts[L]=" ".join(val); j=k; continue
                if opts: break
                if lj.strip() and not num_re.match(lj):
                    stem.append(lj.strip()); j+=1
                else: break
            if len(opts)>=3:
                out[(disc,num)]={"stem":" ".join(stem).strip(),"options":opts}
                i=j; continue
        i+=1
    return out

def parse_answers(a, b):
    out={}  # (disc,num)->letter
    disc=None; i=a
    while i<b:
        s=lines[i].strip()
        if s in DISC:
            disc=s; i+=1; continue
        m=ans_re.match(lines[i])
        if m and disc:
            out[(disc,int(m.group(1)))]=m.group(2)
        i+=1
    return out

questions = parse_questions(q_start, a_start)
answers   = parse_answers(a_start, N)

records=[]
for key,qd in sorted(questions.items()):
    disc,num=key; ans=answers.get(key)
    records.append({"id":f"NBDE-{disc[:4]}-{num}","subject":disc,
                    "stem":qd["stem"],"options":qd["options"],
                    "answer":ans,"n_options":len(qd["options"])})

have=sum(1 for r in records if r["answer"] and r["answer"] in r["options"])
with open("english/nbde.jsonl","w") as f:
    for r in records: f.write(json.dumps(r,ensure_ascii=False)+"\n")
print(f"NBDE q_start={q_start} a_start={a_start}")
print(f"questions parsed : {len(questions)}   answers parsed: {len(answers)}")
print(f"records          : {len(records)}   with valid answer: {have} ({100*have/max(len(records),1):.0f}%)")
print("n_options:",dict(sorted(Counter(r['n_options'] for r in records).items())))
print("answer letters:",dict(sorted(Counter(r['answer'] for r in records if r['answer']).items())))
print("subjects:")
for s,c in Counter(r['subject'] for r in records).most_common(): print(f"   {s:38s} {c}")
