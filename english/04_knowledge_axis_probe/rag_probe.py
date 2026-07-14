#!/usr/bin/env python3
"""RAG upper-bound probe (Axis A): does injecting external dental knowledge help?
Zero training. Base Qwen2.5-14B answers test_ukus (a) closed-book, (b) with top-k retrieved
textbook explanation passages injected. Leakage guard: drop any passage with cosine>0.92 to
the query (that's the test item's own explanation leaking in).
"""
import json, re, sys
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE="/home/student/arthas/mentalDistill/models/Qwen2.5-14B-Instruct"
TEST="english/03_main_distill/data/test_ukus.jsonl"
CORP=["english/04_knowledge_axis_probe/corpus_bof.jsonl",
      "english/04_knowledge_axis_probe/corpus_nbde.jsonl",
      "english/04_knowledge_axis_probe/corpus_medmcqa.jsonl"]
LETTERS=["A","B","C","D","E"]
TOPK=3; LEAK_THRESH=0.92

# BoF corpus file was saved as corpus_bof? we saved as english/04_rag_probe_corpus_bof.jsonl then moved
import os
if not os.path.exists("english/04_knowledge_axis_probe/corpus_bof.jsonl") and \
   os.path.exists("english/04_knowledge_axis_probe/04_rag_probe_corpus_bof.jsonl"):
    CORP[0]="english/04_knowledge_axis_probe/04_rag_probe_corpus_bof.jsonl"

def load(p): return [json.loads(l) for l in open(p)] if os.path.exists(p) else []
corpus=[]
for c in CORP: corpus += load(c)
passages=[d["text"] for d in corpus]
print(f"[corpus] {len(passages)} passages")

test=load(TEST)
print(f"[test] {len(test)} items")

emb=SentenceTransformer("all-MiniLM-L6-v2")
P=emb.encode(passages, normalize_embeddings=True, show_progress_bar=False)

def q_text(r):
    opts=r["Options"]; ol="\n".join(f"{k}. {opts[k]}" for k in LETTERS if k in opts)
    return f"{r['stem'] if 'stem' in r else r['Question']}\n{ol}"

tok=AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
model=AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
model.eval()
opt_ids=[tok.encode(c,add_special_tokens=False)[0] for c in LETTERS]

SYS="You are a dental medicine expert. Answer the single-best-answer question. Output only one capital letter (A/B/C/D/E)."

def ask(qtext, context=None):
    user=qtext
    if context: user=f"Reference material:\n{context}\n\n{qtext}"
    msg=[{"role":"system","content":SYS},{"role":"user","content":user+"\nAnswer with one letter."}]
    text=tok.apply_chat_template(msg,tokenize=False,add_generation_prompt=True)
    ids=tok(text,return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits=model(**ids).logits[0,-1]
    ol=torch.tensor([logits[i].item() for i in opt_ids])
    return LETTERS[int(torch.argmax(ol))]

def q_opts(r): return r["Options"]
def gt(r): return str(r.get("Answer","")).upper()

cb=0; rag=0; n=0
for r in test:
    q=q_text(r); g=gt(r)
    if g not in LETTERS: continue
    n+=1
    # closed book
    if ask(q)==g: cb+=1
    # retrieve
    qv=emb.encode([q],normalize_embeddings=True)[0]
    sims=P@qv
    order=np.argsort(-sims)
    picked=[]
    for idx in order:
        if sims[idx]>LEAK_THRESH: continue   # leakage guard
        picked.append(passages[idx])
        if len(picked)>=TOPK: break
    ctx="\n---\n".join(picked)
    if ask(q,ctx)==g: rag+=1

print(f"\n=== RAG PROBE (n={n}, base 14B, test_ukus) ===")
print(f"closed-book : {100*cb/n:.2f}%")
print(f"+RAG top{TOPK} : {100*rag/n:.2f}%  (delta {100*(rag-cb)/n:+.2f}pp)")
print(f"teacher DeepSeek-V3 ref: 88.30%")
json.dump({"n":n,"closed_book":round(100*cb/n,2),"rag":round(100*rag/n,2),
           "delta_pp":round(100*(rag-cb)/n,2),"topk":TOPK,"leak_thresh":LEAK_THRESH,
           "n_passages":len(passages)},
          open("english/04_knowledge_axis_probe/rag_result.json","w"),indent=2)
