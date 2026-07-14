#!/usr/bin/env python3
"""Self-consistency probe (Axis C): does test-time CoT sampling + majority vote help?
Zero training. base Qwen2.5-14B on test_ukus:
  (a) greedy single letter (closed-book baseline)
  (b) K sampled chain-of-thought reasonings -> extract letter -> majority vote
This is CLOSED-BOOK (no external knowledge) — tests whether more test-time compute alone helps
the knowledge-gap dental questions. Complementary to RAG (which needs a corpus).
"""
import json, os, re, collections
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE="/home/student/arthas/mentalDistill/models/Qwen2.5-14B-Instruct"
TEST="english/03_main_distill/data/test_ukus.jsonl"
LETTERS=["A","B","C","D","E"]; K=5
def load(p): return [json.loads(l) for l in open(p)]
test=load(TEST)

tok=AutoTokenizer.from_pretrained(BASE,trust_remote_code=True)
model=AutoModelForCausalLM.from_pretrained(BASE,dtype=torch.bfloat16,device_map="auto",trust_remote_code=True).eval()
opt_ids=[tok.encode(c,add_special_tokens=False)[0] for c in LETTERS]

def q_text(r):
    o=r["Options"]; ol="\n".join(f"{k}. {o[k]}" for k in LETTERS if k in o)
    return f"{r.get('stem',r.get('Question',''))}\n{ol}"
def gt(r): return str(r.get("Answer","")).upper()

SYS_G="You are a dental medicine expert. Answer the single-best-answer question. Output only one capital letter (A/B/C/D/E)."
SYS_C="You are a dental medicine expert. Think step by step briefly, then end your answer with 'Answer: X' where X is one of A/B/C/D/E."

def greedy(q):
    m=[{"role":"system","content":SYS_G},{"role":"user","content":q+"\nAnswer with one letter."}]
    t=tok.apply_chat_template(m,tokenize=False,add_generation_prompt=True)
    ids=tok(t,return_tensors="pt").to(model.device)
    with torch.no_grad(): lg=model(**ids).logits[0,-1]
    return LETTERS[int(torch.argmax(torch.tensor([lg[i].item() for i in opt_ids])))]

def cot_sample(q):
    m=[{"role":"system","content":SYS_C},{"role":"user","content":q}]
    t=tok.apply_chat_template(m,tokenize=False,add_generation_prompt=True)
    ids=tok(t,return_tensors="pt").to(model.device)
    with torch.no_grad():
        out=model.generate(**ids,max_new_tokens=256,do_sample=True,temperature=0.7,top_p=0.9,
                           pad_token_id=tok.eos_token_id)
    txt=tok.decode(out[0][ids["input_ids"].shape[1]:],skip_special_tokens=True)
    m2=re.findall(r'Answer:\s*([A-E])',txt)
    if m2: return m2[-1]
    m3=re.findall(r'\b([A-E])\b',txt[::-1])  # last standalone letter
    return m3[0] if m3 else None

g_ok=0; sc_ok=0; n=0
for r in test:
    ans=gt(r)
    if ans not in LETTERS: continue
    n+=1
    q=q_text(r)
    if greedy(q)==ans: g_ok+=1
    votes=[]
    for _ in range(K):
        v=cot_sample(q)
        if v in LETTERS: votes.append(v)
    if votes:
        win=collections.Counter(votes).most_common(1)[0][0]
        if win==ans: sc_ok+=1

print(f"=== SELF-CONSISTENCY PROBE (n={n}, base 14B, closed-book, K={K}) ===")
print(f"greedy single      : {100*g_ok/n:.2f}%")
print(f"CoT+majority(K={K}) : {100*sc_ok/n:.2f}%  (delta {100*(sc_ok-g_ok)/n:+.2f}pp)")
print(f"teacher ref 88.30% | RAG(same-domain) ref +11.7~21pp")
json.dump({"n":n,"greedy":round(100*g_ok/n,2),"self_consistency":round(100*sc_ok/n,2),
           "delta_pp":round(100*(sc_ok-g_ok)/n,2),"K":K},
          open("english/04_knowledge_axis_probe/selfconsistency_result.json","w"),indent=2)
