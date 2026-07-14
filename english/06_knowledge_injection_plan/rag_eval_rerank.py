#!/usr/bin/env python3
"""RAG + cross-encoder rerank. Root-cause showed model uses answer-context (66%->95.7%),
so bottleneck is retrieval ranking. Retrieve BGE top-100 -> rerank with cross-encoder -> top-5.
Corpus = answer-type (Dentistry_RAG+StatPearls+Wiki) + PubMed (full, since rerank filters noise).
"""
import json, os
import numpy as np, torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE="/home/student/arthas/mentalDistill/models/Qwen2.5-14B-Instruct"
TEST="english/03_main_distill/data/test_ukus.jsonl"
CORP="english/06_knowledge_injection_plan/corpus"
EMB_MODEL="BAAI/bge-large-en-v1.5"; RERANKER="BAAI/bge-reranker-large"
LETTERS=["A","B","C","D","E"]; POOL=100; TOPK=5

passages=[json.loads(l)["text"] for l in open(f"{CORP}/passages.jsonl")]
P=np.load(f"{CORP}/embeddings.npy")  # already BGE-embedded (dim 1024)
assert P.shape[1]==1024, f"expected BGE 1024-dim, got {P.shape}"
print(f"[corpus] {len(passages)} passages", flush=True)

test=[json.loads(l) for l in open(TEST)]
emb=SentenceTransformer(EMB_MODEL)
def q_text(r):
    o=r["Options"]; ol="\n".join(f"{k}. {o[k]}" for k in LETTERS if k in o)
    return f"{r.get('stem',r.get('Question',''))}\n{ol}"
Q=emb.encode([q_text(r) for r in test], normalize_embeddings=True, show_progress_bar=False)
try:
    reranker=CrossEncoder(RERANKER, max_length=512)
    print(f"[rerank] {RERANKER}", flush=True)
except Exception as e:
    print(f"[rerank] {RERANKER} unavailable: {repr(e)[:80]}; fallback ms-marco", flush=True)
    reranker=CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)

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

cb=0; rr=0; n=0
for qi,r in enumerate(test):
    g=gt(r)
    if g not in LETTERS: continue
    n+=1; q=q_text(r)
    if ask(q)==g: cb+=1
    pool=np.argsort(-(P@Q[qi]))[:POOL]
    pairs=[[q_text(r), passages[i]] for i in pool]
    scores=reranker.predict(pairs, show_progress_bar=False)
    top=[pool[i] for i in np.argsort(-scores)[:TOPK]]
    ctx="\n---\n".join(passages[i] for i in top)
    if ask(q,ctx)==g: rr+=1
print(f"\n=== RAG + RERANK (n={n}, pool{POOL}->top{TOPK}, BGE+cross-encoder) ===")
print(f"closed-book   : {100*cb/n:.2f}%")
print(f"+RAG rerank   : {100*rr/n:.2f}%  (delta {100*(rr-cb)/n:+.2f}pp)")
print(f"teacher 88.30% | inject-answer ceiling 95.7% | prior no-rerank +0~1pp")
json.dump({"n":n,"closed_book":round(100*cb/n,2),"rag_rerank":round(100*rr/n,2),
           "delta_pp":round(100*(rr-cb)/n,2),"pool":POOL,"topk":TOPK},
          open(f"{CORP}/../rag_eval_rerank_result.json","w"),indent=2)
