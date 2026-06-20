#!/usr/bin/env python3
"""prior-art 检索: 用 arXiv API 查两个方向的相关论文。带限流退避。"""
import urllib.request, urllib.parse, time, re, sys

QUERIES = {
  "D2-蒸馏缩放定律": [
    '"distillation scaling laws"',
    'knowledge distillation student model size scaling',
    'compute optimal knowledge distillation',
  ],
  "D2-任务难度感知容量": [
    'task difficulty aware model compression',
    'predict student model size distillation accuracy',
  ],
  "工具-不确定性蒸馏筛选": [
    'uncertainty based knowledge distillation sample selection',
    'confidence based distillation data filtering',
  ],
  "工具-难度与不确定性验证": [
    'data difficulty estimation model uncertainty human agreement',
    'example difficulty annotator agreement model confidence',
  ],
}

def search(q, n=4):
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode({
        "search_query": f"all:{q}", "start": 0, "max_results": n,
        "sortBy": "relevance"})
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent":"research-priorart/1.0"})
            xml = urllib.request.urlopen(req, timeout=30).read().decode("utf-8","ignore")
            entries = re.findall(r"<entry>(.*?)</entry>", xml, re.S)
            out=[]
            for e in entries:
                t = re.search(r"<title>(.*?)</title>", e, re.S)
                idm = re.search(r"<id>(.*?)</id>", e, re.S)
                pub = re.search(r"<published>(.*?)</published>", e, re.S)
                t = re.sub(r"\s+"," ",t.group(1)).strip() if t else "?"
                aid = idm.group(1).strip() if idm else "?"
                yr = pub.group(1)[:4] if pub else "?"
                out.append((yr, t, aid))
            return out
        except Exception as ex:
            if attempt<3: time.sleep(5*(attempt+1)); continue
            return [("ERR",str(ex),"")]
    return []

for topic, qs in QUERIES.items():
    print(f"\n{'='*70}\n# {topic}\n{'='*70}")
    seen=set()
    for q in qs:
        print(f"\n  [query] {q}")
        for yr,t,aid in search(q):
            if aid in seen: continue
            seen.add(aid)
            print(f"    {yr} | {t[:90]}")
            print(f"          {aid}")
        time.sleep(3)
