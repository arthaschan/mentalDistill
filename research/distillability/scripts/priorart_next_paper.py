#!/usr/bin/env python3
"""下一篇论文6个候选方向的 arXiv 查重(分类限定cs.LG/CL/AI)。"""
import urllib.request, urllib.parse, time, re

QUERIES = {
  "A1-TDA拓扑剪枝": 'abs:"persistent homology" AND abs:pruning',
  "A1b-拓扑表征分析": 'abs:topological AND abs:"neural network" AND abs:representation',
  "A2-自适应秩LoRA": 'abs:"adaptive rank" AND abs:LoRA',
  "A2b-Fisher-LoRA": 'abs:Fisher AND abs:LoRA AND abs:rank',
  "A3-各向异性校准": 'abs:anisotropy AND abs:transformer AND abs:representation',
  "B2-step级推理不确定性": 'abs:"step-level" AND abs:uncertainty AND abs:reasoning',
  "B2b-过程奖励不确定": 'abs:"process reward" AND abs:uncertainty',
  "B1-表征指标审计": 'abs:probing AND abs:representation AND (abs:spurious OR abs:correlation)',
  "B5-RoPE长度外推诊断": 'abs:RoPE AND abs:"length extrapolation"',
}

def search(q, n=4):
    url="https://export.arxiv.org/api/query?"+urllib.parse.urlencode({
        "search_query":f"({q}) AND (cat:cs.LG OR cat:cs.CL OR cat:cs.AI)",
        "start":0,"max_results":n,"sortBy":"relevance"})
    for att in range(4):
        try:
            xml=urllib.request.urlopen(urllib.request.Request(url,headers={"User-Agent":"r/1.0"}),timeout=30).read().decode("utf-8","ignore")
            es=re.findall(r"<entry>(.*?)</entry>",xml,re.S); out=[]
            for e in es:
                t=re.search(r"<title>(.*?)</title>",e,re.S)
                pub=re.search(r"<published>(.*?)</published>",e,re.S)
                idm=re.search(r"<id>(.*?)</id>",e,re.S)
                out.append((pub.group(1)[:4] if pub else "?",
                    re.sub(r"\s+"," ",t.group(1)).strip() if t else "?",
                    idm.group(1).strip().split("/")[-1] if idm else ""))
            return out
        except Exception as ex:
            if att<3: time.sleep(5); continue
            return [("ERR",str(ex),"")]
    return []

for topic,q in QUERIES.items():
    print(f"\n{'='*70}\n# {topic}\n{'='*70}")
    r=search(q)
    if not r: print("  (无命中)")
    for yr,t,aid in r:
        print(f"  {yr} | {t[:82]} [{aid}]")
    time.sleep(3)
