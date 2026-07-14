#!/usr/bin/env python3
"""Clean RAG eval on the EXTERNAL dental knowledge base (PubMed+Dentistry_RAG+StatPearls+Wiki).
base Qwen2.5-14B on test_ukus 94: closed-book vs +RAG top-k. Corpus is external (not BoF/NBDE),
so no same-source leakage; we still report max cosine for transparency.
Uses the SAME medical embedding (S-PubMedBert) that built the index.
"""
import json, os, re
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE="/home/student/arthas/mentalDistill/models/Qwen2.5-14B-Instruct"
TEST="english/03_main_distill/data/test_ukus.jsonl"
CORP="english/06_knowledge_injection_plan/corpus"
LETTERS=["A","B","C","D","E"]; TOPK=3
meta=json.load(open(f"{CORP}/index_meta.json"))
EMB_MODEL=meta["model"]

passages=[json.loads(l)["text"] for l in open(f"{CORP}/passages.jsonl")]
P=np.load(f"{CORP}/embeddings.npy")
print(f"[corpus] {len(passages)} passages, emb {P.shape}, model={EMB_MODEL}", flush=True)

test=[json.loads(l) for l in open(TEST)]
emb=SentenceTransformer(EMB_MODEL)
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

cb=0; rag=0; n=0; maxsims=[]
for qi,r in enumerate(test):
    g=gt(r)
    if g not in LETTERS: continue
    n+=1; q=q_text(r)
    if ask(q)==g: cb+=1
    sims=P@Q[qi]; order=np.argsort(-sims)[:TOPK]
    maxsims.append(float(sims[order[0]]))
    ctx="\n---\n".join(passages[i] for i in order)
    if ask(q,ctx)==g: rag+=1

print(f"\n=== CLEAN RAG EVAL (external corpus, n={n}, base 14B) ===")
print(f"closed-book : {100*cb/n:.2f}%")
print(f"+RAG top{TOPK} : {100*rag/n:.2f}%  (delta {100*(rag-cb)/n:+.2f}pp)")
print(f"max-sim: median={np.median(maxsims):.3f} p95={np.percentile(maxsims,95):.3f} (external corpus, no同源)")
print(f"teacher DeepSeek-V3 ref: 88.30%")
json.dump({"n":n,"closed_book":round(100*cb/n,2),"rag":round(100*rag/n,2),
           "delta_pp":round(100*(rag-cb)/n,2),"topk":TOPK,"emb_model":EMB_MODEL,
           "maxsim_median":round(float(np.median(maxsims)),3),"n_passages":len(passages)},
          open(f"{CORP}/../rag_eval_clean_result.json","w"),indent=2)
