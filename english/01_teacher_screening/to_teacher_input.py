#!/usr/bin/env python3
"""Convert unified dental dataset -> teacher-label-generator input schema.
Generator expects: Question, Options{A-E}, Answer, (+ Medical Discipline for domain map).
English data has NO human 'Difficulty level' -> we will use cross-model consensus as the
difficulty gold standard downstream (documented gap, not a bug).
"""
import json, os
SRC="english/dataset/single_best_all.jsonl"
OUT="english/01_teacher_screening/screen_input.jsonl"
os.makedirs(os.path.dirname(OUT),exist_ok=True)
n=0
with open(OUT,"w") as w:
    for line in open(SRC):
        r=json.loads(line)
        rec={
            "uid": r["uid"],
            "Question": r["stem"],
            "Options": r["options"],
            "Answer": r["answer"],
            "Medical Discipline": r["subject"],
            "source": r["source"],
            "n_options": r["n_options"],
        }
        w.write(json.dumps(rec,ensure_ascii=False)+"\n"); n+=1
print(f"wrote {n} -> {OUT}")
