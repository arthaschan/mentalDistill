#!/usr/bin/env python3
"""用 arXiv API (限定 cs.LG/cs.CL 分类, 提高相关性) 补查体检工具方向文献。"""
import urllib.request, urllib.parse, time, re

# arXiv 高级查询: 关键词 AND 限定CS分类
QUERIES = {
  "不确定性蒸馏筛选": 'abs:"knowledge distillation" AND abs:uncertainty AND abs:selection',
  "置信度数据筛选蒸馏": 'abs:distillation AND abs:confidence AND abs:filtering',
  "样本难度估计+不确定性": 'abs:"example difficulty" AND abs:uncertainty',
  "标注者一致性+模型置信": 'abs:"annotator agreement" AND abs:confidence',
  "教师不确定性选择蒸馏": 'abs:teacher AND abs:uncertainty AND abs:distillation',
}

def search(q, n=5):
    url="https://export.arxiv.org/api/query?"+urllib.parse.urlencode({
        "search_query":f"({q}) AND (cat:cs.LG OR cat:cs.CL OR cat:cs.AI)",
        "start":0,"max_results":n,"sortBy":"relevance"})
    for att in range(4):
        try:
            req=urllib.request.Request(url,headers={"User-Agent":"priorart/1.0"})
            xml=urllib.request.urlopen(req,timeout=30).read().decode("utf-8","ignore")
            es=re.findall(r"<entry>(.*?)</entry>",xml,re.S); out=[]
            for e in es:
                t=re.search(r"<title>(.*?)</title>",e,re.S)
                idm=re.search(r"<id>(.*?)</id>",e,re.S)
                pub=re.search(r"<published>(.*?)</published>",e,re.S)
                ab=re.search(r"<summary>(.*?)</summary>",e,re.S)
                out.append((pub.group(1)[:4] if pub else "?",
                    re.sub(r"\s+"," ",t.group(1)).strip() if t else "?",
                    idm.group(1).strip() if idm else "?",
                    re.sub(r"\s+"," ",ab.group(1)).strip()[:170] if ab else ""))
            return out
        except Exception as ex:
            if att<3: time.sleep(5*(att+1)); continue
            return [("ERR",str(ex),"","")]
    return []

for topic,q in QUERIES.items():
    print(f"\n{'='*72}\n# {topic}\n{'='*72}")
    for yr,t,aid,ab in search(q):
        print(f"  {yr} | {t[:88]}")
        print(f"        {aid}")
        if ab: print(f"        摘要: {ab}...")
    time.sleep(3)
