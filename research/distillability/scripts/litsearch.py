#!/usr/bin/env python3
"""Prior-art literature search harness for the distillability-geometry novelty check.
Hits arXiv API (reliable) + Semantic Scholar (with long backoff). Saves JSONL + a
readable report. Designed to run unattended in background."""
import urllib.request, urllib.parse, urllib.error, json, time, re, sys

OUT = "research/distillability/litsearch_results.txt"

# Three claim layers x query battery
QUERIES = {
 "L1_measurement": [
   "Fisher-Rao distance knowledge distillation",
   "information geometry knowledge distillation soft labels",
   "probability simplex distillation teacher geometry",
   "Fisher information metric distillation output distribution",
   "alpha-divergence knowledge distillation",
   "soft label quality measure distillation",
 ],
 "L2_predictive_law": [
   "predicting model correctness from softmax distribution confidence",
   "when does knowledge distillation work teacher student capacity gap",
   "teacher quality distillation effectiveness relationship",
   "confidence separability errors model size scaling",
   "larger teacher not always better distillation",
   "error detection output distribution without ground truth",
 ],
 "L3_algorithm": [
   "sample selection knowledge distillation soft label filtering",
   "training-free data valuation distillation",
   "confidence based sample weighting distillation",
   "noisy teacher label filtering distillation",
   "per-sample teacher selection routing distillation",
   "selective distillation reliable samples",
 ],
}

def arxiv(query, n=8):
    url="http://export.arxiv.org/api/query?"+urllib.parse.urlencode(
        {"search_query":f"all:{query}","start":0,"max_results":n,"sortBy":"relevance"})
    try:
        with urllib.request.urlopen(url,timeout=30) as r: data=r.read().decode()
    except Exception as e: return [f"ARXIV ERROR: {e}"]
    out=[]
    for e in re.findall(r"<entry>(.*?)</entry>",data,re.S):
        t=re.search(r"<title>(.*?)</title>",e,re.S)
        s=re.search(r"<summary>(.*?)</summary>",e,re.S)
        p=re.search(r"<published>(.*?)</published>",e,re.S)
        t=t.group(1).strip().replace("\n"," ") if t else "?"
        yr=p.group(1)[:4] if p else "?"
        s=(s.group(1).strip().replace("\n"," ")[:240]) if s else ""
        out.append(f"  [{yr}] {t}\n        {s}")
    return out or ["  (no arxiv results)"]

def ss(query,n=8,tries=5):
    base="https://api.semanticscholar.org/graph/v1/paper/search"
    url=f"{base}?"+urllib.parse.urlencode({"query":query,"limit":n,
        "fields":"title,year,venue,abstract,citationCount"})
    for t in range(tries):
        try:
            req=urllib.request.Request(url,headers={"User-Agent":"novelty-check"})
            with urllib.request.urlopen(req,timeout=30) as r:
                res=json.loads(r.read().decode()); out=[]
                for p in res.get("data",[])[:n]:
                    ab=(p.get('abstract') or '')[:240].replace("\n"," ")
                    out.append(f"  [{p.get('year')}] {p.get('title')} "
                               f"({p.get('venue') or '?'}; cites={p.get('citationCount')})\n        {ab}")
                return out or ["  (no ss results)"]
        except urllib.error.HTTPError as e:
            if e.code==429: time.sleep(8*(t+1)); continue
            return [f"  SS ERROR {e.code}"]
        except Exception: time.sleep(4)
    return ["  SS rate-limited after retries"]

def main():
    with open(OUT,"w",encoding="utf-8") as f:
        f.write("PRIOR-ART SEARCH for distillability-geometry novelty\n")
        f.write(f"generated {time.strftime('%Y-%m-%d %H:%M')}\n\n")
        for layer,qs in QUERIES.items():
            f.write("#"*88+f"\n# {layer}\n"+"#"*88+"\n")
            for q in qs:
                f.write(f"\n=== QUERY: {q} ===\n")
                f.write("-- arXiv --\n"+"\n".join(arxiv(q))+"\n")
                f.flush(); time.sleep(3)
                f.write("-- Semantic Scholar --\n"+"\n".join(ss(q))+"\n")
                f.flush(); time.sleep(6)
        f.write("\n[DONE]\n")
    print(f"saved {OUT}")

if __name__=="__main__":
    main()
