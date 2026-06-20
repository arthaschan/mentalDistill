#!/usr/bin/env python3
"""查重: '分析LLM空间结构→提升Transformer训练速度/质量' 这个方向的主要流派。"""
import urllib.request, urllib.parse, time, re

QUERIES = {
  "训练动力学几何分析": 'abs:"training dynamics" AND abs:geometry AND abs:transformer',
  "loss landscape几何": 'abs:"loss landscape" AND abs:"neural network" AND (abs:curvature OR abs:sharpness)',
  "表征几何指导训练": 'abs:representation AND abs:geometry AND abs:training AND abs:"language model"',
  "神经坍缩": 'abs:"neural collapse" AND abs:training',
  "训练加速架构/初始化": 'abs:transformer AND (abs:"faster training" OR abs:"training efficiency") AND (abs:initialization OR abs:normalization)',
  "几何感知优化器": 'abs:geometry AND abs:optimizer AND abs:"deep learning"',
  "内在维度训练": 'abs:"intrinsic dimension" AND abs:training AND abs:"neural network"',
  "信息几何深度学习训练": 'abs:"information geometry" AND abs:training AND abs:"neural network"',
  "表征坍缩诊断干预": 'abs:representation AND (abs:collapse OR abs:rank) AND abs:transformer AND abs:training',
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
                out.append((pub.group(1)[:7] if pub else "?",
                    re.sub(r"\s+"," ",t.group(1)).strip() if t else "?",
                    idm.group(1).strip().split("/")[-1] if idm else ""))
            return out
        except Exception as ex:
            if att<3: time.sleep(5); continue
            return [("ERR",str(ex),"")]
    return []

for topic,q in QUERIES.items():
    print(f"\n{'='*72}\n# {topic}\n{'='*72}")
    for yr,t,aid in search(q):
        print(f"  {yr} | {t[:80]} [{aid}]")
    time.sleep(2)
