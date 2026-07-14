#!/usr/bin/env python3
"""Merge dental corpora -> dedup -> chunk -> embed -> save index.
Tries a medical embedding (pritamdeka/S-PubMedBert-MS-MARCO) first; falls back to all-MiniLM-L6-v2.
Outputs: corpus/passages.jsonl (chunked text), corpus/embeddings.npy, corpus/index_meta.json
"""
import json, os, re, sys
import numpy as np

CORPDIR="english/06_knowledge_injection_plan/corpus"
SRCS=[f"{CORPDIR}/corpus_pubmed.jsonl", f"{CORPDIR}/corpus_hf.jsonl"]
# optional user textbooks: any file placed in corpus/user_*.jsonl with a "text" field
import glob
SRCS += sorted(glob.glob(f"{CORPDIR}/user_*.jsonl"))

def clean(s): return re.sub(r'\s+',' ',str(s)).strip()

# 1. merge + dedup + chunk (~256 words, no overlap; split long abstracts)
seen=set(); passages=[]
for src in SRCS:
    if not os.path.exists(src): continue
    for line in open(src):
        try: r=json.loads(line)
        except: continue
        t=clean(r.get("text","")); srcname=r.get("src","?")
        if len(t)<120: continue
        words=t.split()
        # chunk into <=256-word pieces
        for i in range(0,len(words),256):
            chunk=" ".join(words[i:i+256])
            if len(chunk)<120: continue
            key=chunk[:100].lower()
            if key in seen: continue
            seen.add(key)
            passages.append({"src":srcname,"text":chunk})
print(f"[merge] {len(passages)} chunks from {len([s for s in SRCS if os.path.exists(s)])} sources", flush=True)
with open(f"{CORPDIR}/passages.jsonl","w",encoding="utf-8") as w:
    for p in passages: w.write(json.dumps(p,ensure_ascii=False)+"\n")

# 2. embed
from sentence_transformers import SentenceTransformer
model_name=None
for cand in ["BAAI/bge-large-en-v1.5","BAAI/bge-base-en-v1.5","pritamdeka/S-PubMedBert-MS-MARCO","all-MiniLM-L6-v2"]:
    try:
        emb=SentenceTransformer(cand); model_name=cand; break
    except Exception as e:
        print(f"[embed] {cand} unavailable: {repr(e)[:80]}", flush=True)
print(f"[embed] using {model_name}", flush=True)
texts=[p["text"] for p in passages]
E=emb.encode(texts, normalize_embeddings=True, batch_size=128, show_progress_bar=True)
np.save(f"{CORPDIR}/embeddings.npy", E.astype(np.float32))
json.dump({"model":model_name,"n":len(passages),"dim":int(E.shape[1]),
           "src_counts":{s:sum(1 for p in passages if p["src"]==s) for s in set(p["src"] for p in passages)}},
          open(f"{CORPDIR}/index_meta.json","w"),indent=2)
print(f"[done] {len(passages)} passages, dim={E.shape[1]}, model={model_name}", flush=True)
print("src counts:", {s:sum(1 for p in passages if p['src']==s) for s in set(p['src'] for p in passages)})
