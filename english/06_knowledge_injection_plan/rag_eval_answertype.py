#!/usr/bin/env python3
"""RAG eval with ANSWER-TYPE corpus only (Dentistry_RAG + StatPearls, drop PubMed research
abstracts) + top-5 + BGE embedding. Tests if removing research-narrative noise + more context
converts the 86% oracle ceiling into real gain.
"""
import json, os, re
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE="/home/student/arthas/mentalDistill/models/Qwen2.5-14B-Instruct"
TEST="english/03_main_distill/data/test_ukus.jsonl"
CORP="english/06_knowledge_injection_plan/corpus"
EMB_MODEL="BAAI/bge-large-en-v1.5"
LETTERS=["A","B","C","D","E"]; TOPK=5
KEEP={"Dentistry_RAG","StatPearls","Wikipedia"}  # drop PubMed

# filter answer-type passages
passages=[]
for l in open(f"{CORP}/passages.jsonl"):
    r=json.loads(l)
    if r["src"] in KEEP: passages.append(r["text"])
print(f"[corpus] answer-type passages: {len(passages)} (kept {KEEP})", flush=True)

emb=SentenceTransformer(EMB_MODEL)
P=emb.encode(passages, normalize_embeddings=True, batch_size=128, show_progress_bar=False)

test=[json.loads(l) for l in open(TEST)]
def q_text(r):
    o=r["Options"]; ol="\n".join(f"{k}. {o[k]}" for k in LETTERS if k in o)
    return f"{r.get('stem',r.get('Question',''))}\n{ol}"
Q=emb.encode([q_text(r) for r in test], normalize_embeddings=True, show_progress_bar=False)

tok=AutoTokenizer.from_pretrained(BASE,trust_remote_code=True)
model=AutoModelForCausalLM.from_pretrained(BASE,dtype=torch.bfloat16,device_map="auto",trust_remote_code=True).eval()
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

cb=0; rag=0; n=0
for qi,r in enumerate(test):
    g=gt(r)
    if g not in LETTERS: continue
    n+=1; q=q_text(r)
    if ask(q)==g: cb+=1
    order=np.argsort(-(P@Q[qi]))[:TOPK]
    ctx="\n---\n".join(passages[i] for i in order)
    if ask(q,ctx)==g: rag+=1
print(f"\n=== ANSWER-TYPE RAG (n={n}, base 14B, top{TOPK}, BGE) ===")
print(f"closed-book : {100*cb/n:.2f}%")
print(f"+RAG        : {100*rag/n:.2f}%  (delta {100*(rag-cb)/n:+.2f}pp)")
print(f"teacher 88.30% | prior: PubMed-mix top3 +1.06pp / oracle ceiling 86%")
json.dump({"n":n,"closed_book":round(100*cb/n,2),"rag":round(100*rag/n,2),
           "delta_pp":round(100*(rag-cb)/n,2),"topk":TOPK,"corpus":"answer-type","n_passages":len(passages)},
          open(f"{CORP}/../rag_eval_answertype_result.json","w"),indent=2)
