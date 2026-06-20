#!/usr/bin/env python3
"""Prior-art literature search harness for the distillability-geometry novelty check.
Hits arXiv API (reliable) + Semantic Scholar (with long backoff). Saves JSONL + a
readable report. Designed to run unattended in background."""
import urllib.request, urllib.parse, urllib.error, json, time, re, sys

OUT = "research/distillability/litsearch_results.txt"

# 第二轮检索: 重心扩到 "training-free 教师/源选择 + 迁移性指标对标 + 表征探针"
# 输出到独立文件, 不覆盖第一轮的几何检索结果.
OUT2 = "research/distillability/litsearch_round2_results.txt"

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


# 第二轮查询电池: training-free 教师/源选择 + 迁移性指标 + 表征探针
QUERIES2 = {
 "T1_training_free_teacher_selection": [
   "training-free teacher selection knowledge distillation",
   "predicting distillation performance without training student",
   "which teacher to distill from selection",
   "distillation gain prediction distillability metric",
   "estimate knowledge distillation effectiveness a priori",
   "teacher ranking distillation before training",
 ],
 "T2_transferability_estimation": [
   "transferability estimation LogME source model selection",
   "LEEP transferability metric pretrained model selection",
   "TransRate H-score transferability transfer learning",
   "transferability estimation knowledge distillation teacher",
   "model selection without fine-tuning transferability score",
 ],
 "T3_teacher_quality_vs_gain": [
   "teacher accuracy distillation student performance relationship",
   "stronger teacher worse student capacity gap distillation",
   "calibration error teacher selection distillation",
   "expected calibration error predict transfer accuracy",
 ],
 "T6_representation_probe": [
   "linear probe predict model correctness hidden states",
   "CKA representation similarity distillation transfer",
   "representation geometry knowledge distillation layer probing",
 ],
}


def main2():
    with open(OUT2,"w",encoding="utf-8") as f:
        f.write("PRIOR-ART SEARCH round 2: training-free teacher selection + transferability metrics\n")
        f.write(f"generated {time.strftime('%Y-%m-%d %H:%M')}\n\n")
        for layer,qs in QUERIES2.items():
            f.write("#"*88+f"\n# {layer}\n"+"#"*88+"\n")
            for q in qs:
                f.write(f"\n=== QUERY: {q} ===\n")
                f.write("-- arXiv --\n"+"\n".join(arxiv(q))+"\n")
                f.flush(); time.sleep(3)
                f.write("-- Semantic Scholar --\n"+"\n".join(ss(q))+"\n")
                f.flush(); time.sleep(6)
        f.write("\n[DONE]\n")
    print(f"saved {OUT2}")


if __name__=="__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "round1"
    if mode == "round2":
        main2()
    else:
        main()
