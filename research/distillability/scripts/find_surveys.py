#!/usr/bin/env python3
"""标准流程第1步: 找 G-a(几何指标祛魅) 和 M1(医疗LLM可靠性) 方向的最新综述。"""
import urllib.request, urllib.parse, time, re

QUERIES = {
  "综述-表征几何/探针": '(ti:survey OR ti:review) AND abs:representation AND (abs:geometry OR abs:probing OR abs:"intrinsic dimension")',
  "综述-模型压缩剪枝": '(ti:survey OR ti:review) AND (abs:pruning OR abs:compression) AND abs:"large language"',
  "综述-医疗LLM评估": '(ti:survey OR ti:review) AND abs:medical AND abs:"large language model" AND (abs:evaluation OR abs:reliability OR abs:trustworth)',
  "综述-LLM校准不确定性": '(ti:survey OR ti:review) AND abs:"large language model" AND (abs:calibration OR abs:uncertainty)',
  "综述-知识蒸馏LLM": '(ti:survey OR ti:review) AND abs:"knowledge distillation" AND abs:"language model"',
  "综述-LLM可解释性机制": '(ti:survey OR ti:review) AND abs:"language model" AND (abs:interpretability OR abs:"mechanistic")',
}

def search(q, n=5):
    url="https://export.arxiv.org/api/query?"+urllib.parse.urlencode({
        "search_query":f"({q}) AND (cat:cs.LG OR cat:cs.CL OR cat:cs.AI)",
        "start":0,"max_results":n,"sortBy":"submittedDate","sortOrder":"descending"})
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
    time.sleep(3)
