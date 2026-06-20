#!/usr/bin/env python3
"""用 Semantic Scholar 免费API补查"体检工具方向"文献(不确定性蒸馏筛选+难度验证)。带限流退避。"""
import urllib.request, urllib.parse, json, time

QUERIES = [
  "uncertainty based knowledge distillation sample selection",
  "confidence based data filtering knowledge distillation",
  "data difficulty estimation model uncertainty human agreement",
  "example difficulty annotator agreement model confidence",
  "teacher uncertainty selective knowledge distillation",
]

def search(q, n=5):
    url="https://api.semanticscholar.org/graph/v1/paper/search?"+urllib.parse.urlencode({
        "query":q,"limit":n,"fields":"title,year,citationCount,abstract"})
    for att in range(5):
        try:
            req=urllib.request.Request(url,headers={"User-Agent":"priorart/1.0"})
            d=json.loads(urllib.request.urlopen(req,timeout=30).read().decode("utf-8","ignore"))
            return d.get("data",[])
        except Exception as ex:
            code=getattr(ex,'code',None)
            if code==429 or att<4:
                time.sleep(8*(att+1)); continue
            return [{"title":f"ERR {ex}","year":"","citationCount":0}]
    return []

for q in QUERIES:
    print(f"\n{'='*72}\n[query] {q}\n{'='*72}")
    for p in search(q):
        t=p.get("title","?"); yr=p.get("year","?"); cc=p.get("citationCount",0)
        ab=(p.get("abstract") or "")[:160].replace("\n"," ")
        print(f"  {yr} | cit={cc} | {t}")
        if ab: print(f"        {ab}...")
    time.sleep(6)
