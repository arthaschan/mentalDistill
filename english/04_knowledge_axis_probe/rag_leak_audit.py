#!/usr/bin/env python3
"""Leakage audit for the RAG probe: the +21pp gain is suspiciously close to teacher.
Because BoF/NBDE test questions come from the SAME books as the corpus, a retrieved
passage may be the test item's OWN explanation (verbatim answer). We:
 1. Report max cosine sim distribution (how many queries have a near-duplicate passage).
 2. Re-run accuracy at STRICTER leak thresholds (0.85, 0.75, 0.6) and with MedMCQA-ONLY
    corpus (different source => cannot contain UK/US test answers) as the clean condition.
"""
import json, os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE="/home/student/arthas/mentalDistill/models/Qwen2.5-14B-Instruct"
TEST="english/03_main_distill/data/test_ukus.jsonl"
LETTERS=["A","B","C","D","E"]; TOPK=3
def load(p): return [json.loads(l) for l in open(p)] if os.path.exists(p) else []

test=load(TEST)
corpora={
 "ALL": load("english/04_knowledge_axis_probe/corpus_bof.jsonl")
       +load("english/04_knowledge_axis_probe/corpus_nbde.jsonl")
       +load("english/04_knowledge_axis_probe/corpus_medmcqa.jsonl"),
 "MedMCQA_only": load("english/04_knowledge_axis_probe/corpus_medmcqa.jsonl"),  # different source, clean
}
emb=SentenceTransformer("all-MiniLM-L6-v2")
def q_text(r):
    o=r["Options"]; ol="\n".join(f"{k}. {o[k]}" for k in LETTERS if k in o)
    return f"{r.get('stem',r.get('Question',''))}\n{ol}"
Q=emb.encode([q_text(r) for r in test],normalize_embeddings=True,show_progress_bar=False)

tok=AutoTokenizer.from_pretrained(BASE,trust_remote_code=True)
model=AutoModelForCausalLM.from_pretrained(BASE,torch_dtype=torch.bfloat16,device_map="auto",trust_remote_code=True).eval()
opt_ids=[tok.encode(c,add_special_tokens=False)[0] for c in LETTERS]
SYS="You are a dental medicine expert. Answer the single-best-answer question. Output only one capital letter (A/B/C/D/E)."
def ask(q,ctx=None):
    u=q if not ctx else f"Reference material:\n{ctx}\n\n{q}"
    m=[{"role":"system","content":SYS},{"role":"user","content":u+"\nAnswer with one letter."}]
    t=tok.apply_chat_template(m,tokenize=False,add_generation_prompt=True)
    ids=tok(t,return_tensors="pt").to(model.device)
    with torch.no_grad(): lg=model(**ids).logits[0,-1]
    return LETTERS[int(torch.argmax(torch.tensor([lg[i].item() for i in opt_ids])))]
def gt(r): return str(r.get("Answer","")).upper()

results={}
for cname,corpus in corpora.items():
    P=emb.encode([d["text"] for d in corpus],normalize_embeddings=True,show_progress_bar=False)
    passages=[d["text"] for d in corpus]
    sims_all=Q@P.T
    maxsim=sims_all.max(axis=1)
    if cname=="ALL":
        results["_maxsim_dist"]={"gt_0.92":int((maxsim>0.92).sum()),"gt_0.85":int((maxsim>0.85).sum()),
                                 "gt_0.75":int((maxsim>0.75).sum()),"median":round(float(np.median(maxsim)),3)}
    for thr in [0.92,0.85,0.75,0.60]:
        c=0;n=0
        for qi,r in enumerate(test):
            g=gt(r)
            if g not in LETTERS: continue
            n+=1
            order=np.argsort(-sims_all[qi]); picked=[]
            for idx in order:
                if sims_all[qi][idx]>thr: continue
                picked.append(passages[idx])
                if len(picked)>=TOPK: break
            if ask(q_text(r),"\n---\n".join(picked))==g: c+=1
        results[f"{cname}@thr{thr}"]=round(100*c/n,2)
        print(f"{cname} leak_thr={thr}: {100*c/n:.2f}%  (n={n})")
json.dump(results,open("english/04_knowledge_axis_probe/rag_leak_audit.json","w"),indent=2)
print("\nmaxsim dist:",results.get("_maxsim_dist"))
