#!/usr/bin/env python3
"""B1/B2 二轮精准查重 + 抓最相关论文摘要。"""
import urllib.request, urllib.parse, time, re

# 针对精确claim的查询
QUERIES = {
  # B1: 表征几何指标是否只是深度/参数量/熵的代理 —— 元审计/伪相关
  "B1-几何指标元分析": 'abs:"intrinsic dimension" AND abs:"layer depth"',
  "B1-探针指标批判": 'abs:probing AND abs:"control task" AND abs:representation',
  "B1-表征度量是否预测能力": 'abs:representation AND abs:metric AND abs:"predictive of" AND abs:performance',
  "B1-几何指标confound": 'abs:geometry AND abs:representation AND abs:confound',
  # B2: 用外部金标准(人类step难度/跨模型step共识)验证step不确定性=客观难度
  "B2-step不确定性vs人类": 'abs:step AND abs:uncertainty AND abs:human AND abs:reasoning',
  "B2-推理步难度标注": 'abs:reasoning AND abs:step AND abs:difficulty AND abs:annotation',
  "B2-PRM与人类对齐": 'abs:"process reward" AND abs:human AND abs:alignment',
  "B2-跨模型step共识": 'abs:reasoning AND abs:step AND abs:agreement AND abs:models',
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
                ab=re.search(r"<summary>(.*?)</summary>",e,re.S)
                out.append((pub.group(1)[:4] if pub else "?",
                    re.sub(r"\s+"," ",t.group(1)).strip() if t else "?",
                    idm.group(1).strip().split("/")[-1] if idm else "",
                    re.sub(r"\s+"," ",ab.group(1)).strip()[:200] if ab else ""))
            return out
        except Exception as ex:
            if att<3: time.sleep(5); continue
            return [("ERR",str(ex),"","")]
    return []

for topic,q in QUERIES.items():
    print(f"\n{'='*72}\n# {topic}\n{'='*72}")
    r=search(q)
    if not r: print("  (无命中)")
    for yr,t,aid,ab in r:
        print(f"  {yr} | {t[:80]} [{aid}]")
        if ab: print(f"      {ab[:150]}...")
    time.sleep(3)
