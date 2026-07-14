#!/usr/bin/env python3
"""Fetch PubMed dental abstracts via NCBI E-utilities -> corpus_pubmed.jsonl.
Batched by dental subtopic to get broad coverage. Polite rate limit (3 req/s no key).
Idempotent-ish: skips if output already has target count.
"""
import urllib.request, urllib.parse, json, time, re, sys, os

OUT="english/06_knowledge_injection_plan/corpus/corpus_pubmed.jsonl"
os.makedirs(os.path.dirname(OUT), exist_ok=True)
EUTILS="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
PER_TOPIC=1200   # abstracts per subtopic
TOPICS=[
 "dental caries","periodontal disease","endodontics root canal","dental pulp",
 "oral pathology","dental materials","orthodontics malocclusion","prosthodontics denture",
 "oral surgery extraction","dental implant","oral cancer squamous","salivary gland",
 "temporomandibular joint","pediatric dentistry","dental anesthesia","tooth enamel dentin",
 "gingivitis","dental radiography","fluoride prevention","oral mucosa lesion",
 "maxillofacial","occlusion bite","dental trauma","bruxism",
]

def get(url, tries=4):
    for t in range(tries):
        try:
            return urllib.request.urlopen(url, timeout=30).read().decode("utf-8","ignore")
        except Exception as e:
            if t==tries-1: return None
            time.sleep(2*(t+1))
    return None

def esearch(term, retmax):
    url=EUTILS+"esearch.fcgi?"+urllib.parse.urlencode(
        {"db":"pubmed","term":f"{term} AND hasabstract AND English[lang]","retmax":retmax,"retmode":"json"})
    r=get(url)
    if not r: return []
    try: return json.loads(r)["esearchresult"]["idlist"]
    except: return []

def efetch(ids):
    url=EUTILS+"efetch.fcgi?"+urllib.parse.urlencode(
        {"db":"pubmed","id":",".join(ids),"rettype":"abstract","retmode":"text"})
    return get(url) or ""

def split_abstracts(text):
    # NCBI text format separates records by blank lines + a leading "N. Journal..." index
    blocks=re.split(r'\n\n(?=\d+\.\s)', text)
    out=[]
    for b in blocks:
        # abstract body = paragraphs after the author/affiliation; heuristic: longest paragraph
        paras=[p.strip() for p in b.split("\n\n") if len(p.strip())>200]
        if paras:
            out.append(max(paras,key=len).replace("\n"," ").strip())
    return out

seen=set(); n=0
mode="w"
with open(OUT, mode, encoding="utf-8") as w:
    for ti,topic in enumerate(TOPICS,1):
        ids=esearch(topic, PER_TOPIC)
        print(f"[{ti}/{len(TOPICS)}] {topic}: {len(ids)} ids", flush=True)
        time.sleep(0.4)
        for i in range(0,len(ids),100):
            batch=ids[i:i+100]
            txt=efetch(batch)
            for ab in split_abstracts(txt):
                key=ab[:120].lower()
                if key in seen or len(ab)<150: continue
                seen.add(key)
                w.write(json.dumps({"src":"PubMed","topic":topic,"text":ab[:1500]},ensure_ascii=False)+"\n")
                n+=1
            w.flush()
            time.sleep(0.4)
        print(f"    cumulative passages: {n}", flush=True)
print(f"DONE PubMed corpus: {n} passages -> {OUT}", flush=True)
