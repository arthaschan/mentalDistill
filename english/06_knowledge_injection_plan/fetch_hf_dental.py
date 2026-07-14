#!/usr/bin/env python3
"""Load HF dental knowledge sources -> corpus_hf.jsonl.
 - Mahmood1998/Dentistry_RAG : dental-specific chunks (nested subchunks[].text)
 - awinml/statpearls        : clinical articles, keep only dental (keyword filter)
 - Wikipedia Category:Dentistry : article extracts via API
"""
import json, re, os, urllib.request, urllib.parse, time
from datasets import load_dataset

OUT="english/06_knowledge_injection_plan/corpus/corpus_hf.jsonl"
os.makedirs(os.path.dirname(OUT), exist_ok=True)
DENT=re.compile(r'\b(tooth|teeth|dental|dentin|enamel|pulp|molar|premolar|incisor|gingiv|periodont|caries|occlus|denture|endodont|orthodont|amalgam|prosthodont|alveolar|odonto|periapical|dentition|malocclusion|oral mucosa|TMJ|temporomandibular|salivary|maxillofacial|gingivitis)\b',re.I)
def clean(s): return re.sub(r'\s+',' ',str(s)).strip()

seen=set(); n=0
w=open(OUT,"w",encoding="utf-8")

# 1. Dentistry_RAG (all dental already)
try:
    ds=load_dataset("Mahmood1998/Dentistry_RAG",split="train",streaming=True)
    c=0
    for row in ds:
        ch=row.get("chunks",{})
        subs=ch.get("subchunks",[]) if isinstance(ch,dict) else []
        for s in subs:
            t=clean(s.get("text",""))
            if len(t)<120: continue
            k=t[:120].lower()
            if k in seen: continue
            seen.add(k); w.write(json.dumps({"src":"Dentistry_RAG","text":t[:1500]},ensure_ascii=False)+"\n"); n+=1; c+=1
    print(f"Dentistry_RAG: {c} passages", flush=True)
except Exception as e: print("Dentistry_RAG ERR",repr(e)[:120], flush=True)

# 2. StatPearls dental-filtered
try:
    ds=load_dataset("awinml/statpearls",split="train",streaming=True)
    c=0
    for row in ds:
        t=clean(row.get("contents",""))
        if not DENT.search(t) or len(t)<150: continue
        # require >=2 distinct dental terms to cut false positives (e.g. 'mouthpiece')
        if len(set(m.lower() for m in DENT.findall(t)))<2: continue
        k=t[:120].lower()
        if k in seen: continue
        seen.add(k); w.write(json.dumps({"src":"StatPearls","text":t[:1500]},ensure_ascii=False)+"\n"); n+=1; c+=1
    print(f"StatPearls dental: {c} passages", flush=True)
except Exception as e: print("StatPearls ERR",repr(e)[:120], flush=True)

# 3. Wikipedia Category:Dentistry article extracts
def wiki_get(params):
    req=urllib.request.Request("https://en.wikipedia.org/w/api.php?"+urllib.parse.urlencode(params),
        headers={"User-Agent":"academic-research/1.0"})
    return json.loads(urllib.request.urlopen(req,timeout=25).read())
try:
    members=[]
    for cat in ["Category:Dentistry","Category:Dental_anatomy","Category:Oral_pathology",
                "Category:Restorative_dentistry","Category:Periodontology","Category:Endodontics",
                "Category:Orthodontics","Category:Prosthodontics"]:
        try:
            r=wiki_get({"action":"query","list":"categorymembers","cmtitle":cat,"cmlimit":"200","cmtype":"page","format":"json"})
            members+=[m["title"] for m in r.get("query",{}).get("categorymembers",[])]
            time.sleep(0.3)
        except: pass
    members=list(dict.fromkeys(members))
    c=0
    for i in range(0,len(members),20):
        batch=members[i:i+20]
        try:
            r=wiki_get({"action":"query","prop":"extracts","exintro":"1","explaintext":"1",
                        "titles":"|".join(batch),"format":"json"})
            for pg in r.get("query",{}).get("pages",{}).values():
                t=clean(pg.get("extract",""))
                if len(t)<150: continue
                k=t[:120].lower()
                if k in seen: continue
                seen.add(k); w.write(json.dumps({"src":"Wikipedia","text":t[:1500]},ensure_ascii=False)+"\n"); n+=1; c+=1
            time.sleep(0.3)
        except: pass
    print(f"Wikipedia dental: {c} passages (from {len(members)} articles)", flush=True)
except Exception as e: print("Wikipedia ERR",repr(e)[:120], flush=True)

w.close()
print(f"DONE HF corpus: {n} passages -> {OUT}", flush=True)
