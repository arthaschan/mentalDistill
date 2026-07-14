#!/usr/bin/env python3
"""Prior-art battery for the English-dental-distillation paper. Writes results to a file
as it goes (flush per query) so rate-limit gaps don't lose data.
Claim layers:
  L1 choice-head / logit distillation on English dental/medical MCQ (student>=teacher)
  L2 multi-teacher fusion NEGATIVE result: complementarity exists but not cheaply exploitable
  L3 teacher entropy vs cross-model-consensus difficulty (external validation), medical MCQ
"""
import urllib.request, urllib.parse, re, time, json, sys

OUT="english/priorart_results.txt"
def log(s):
    with open(OUT,"a") as f: f.write(s+"\n"); f.flush()
    print(s,flush=True)

def arxiv(q, n=6):
    base="https://export.arxiv.org/api/query?"
    p={"search_query":q,"max_results":n,"sortBy":"relevance"}
    url=base+urllib.parse.urlencode(p)
    try:
        raw=urllib.request.urlopen(url,timeout=30).read().decode("utf-8","ignore")
    except Exception as e:
        return [f"[arxiv error: {e}]"]
    entries=re.findall(r'<entry>(.*?)</entry>', raw, re.S)
    out=[]
    for e in entries:
        t=re.search(r'<title>(.*?)</title>', e, re.S)
        y=re.search(r'<published>(\d{4})', e)
        t=re.sub(r'\s+',' ',t.group(1)).strip() if t else "?"
        out.append(f"({y.group(1) if y else '????'}) {t}")
    return out or ["[no hits]"]

def s2(q, n=6, tries=5):
    url=("https://api.semanticscholar.org/graph/v1/paper/search?"
         +urllib.parse.urlencode({"query":q,"limit":n,"fields":"title,year,venue,citationCount"}))
    for t in range(tries):
        try:
            raw=urllib.request.urlopen(url,timeout=30).read().decode("utf-8","ignore")
            data=json.loads(raw).get("data",[])
            return [f"({d.get('year','?')}) {d.get('title','?')} [{d.get('venue','')}|cit={d.get('citationCount',0)}]" for d in data] or ["[no hits]"]
        except Exception as e:
            if "429" in str(e) and t<tries-1:
                time.sleep(8*(t+1)); continue
            return [f"[s2 error: {e}]"]
    return ["[s2 rate-limited after retries]"]

QUERIES={
 "L1_choicehead_dental":[
   'abs:"knowledge distillation" AND abs:dental AND abs:"multiple choice"',
   'all:knowledge distillation dental exam student teacher',
   'abs:"medical" AND abs:"multiple choice" AND abs:distillation AND (cat:cs.CL OR cat:cs.LG)',
   'all:choice head distillation medical MCQ student surpasses teacher',
 ],
 "L2_fusion_negative":[
   'abs:"multi-teacher" AND abs:distillation AND abs:"knowledge distillation" AND (cat:cs.LG OR cat:cs.CL)',
   'all:multi teacher knowledge distillation ensemble does not help negative',
   'abs:"teacher selection" AND abs:distillation AND abs:"which teacher"',
   'all:teacher complementarity routing distillation medical domain',
 ],
 "L3_entropy_consensus_difficulty":[
   'abs:uncertainty AND abs:"question difficulty" AND abs:"large language model"',
   'all:LLM entropy question difficulty cross model consensus agreement',
   'abs:"model uncertainty" AND abs:"human difficulty" AND (cat:cs.CL OR cat:cs.LG)',
   'all:teacher confidence difficulty medical exam distillation label-free',
 ],
}
LANDMARKS=['Does Knowledge Distillation Really Work','Strong Teacher is Not Necessary',
           'confidence-gated distillation','distillation scaling laws']

log(f"===== PRIOR-ART BATTERY  {time.strftime('%Y-%m-%d %H:%M')} =====")
for layer,qs in QUERIES.items():
    log(f"\n############## {layer} ##############")
    for q in qs:
        log(f"\n--- Q: {q}")
        log("[arXiv]");  [log("  "+h) for h in arxiv(q)]
        time.sleep(1)
        log("[S2]");     [log("  "+h) for h in s2(q)]
        time.sleep(6)
log("\n############## LANDMARK title checks ##############")
for q in LANDMARKS:
    log(f"\n--- landmark: {q}")
    [log("  "+h) for h in s2(q,n=3)]
    time.sleep(6)
log("\n===== DONE =====")
