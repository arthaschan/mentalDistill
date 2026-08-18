#!/usr/bin/env python3
"""核实参考文献真实性：用 arxiv.org/abs/{id} 页面（citation_title meta）核对，逐条限速。"""
import re, urllib.request, time, json

CANDIDATES = [
    ("1503.02531", "Distilling the Knowledge in a Neural Network"),
    ("2009.13081", "What Disease Does This Patient Have"),
    ("2203.14371", "MedMCQA"),
    ("2009.03300", "Measuring Massive Multitask Language Understanding"),
    ("1909.06146", "PubMedQA"),
    ("2306.03030", "CMExam"),
    ("2508.20416", "DentalBench"),
    ("2306.12079", "M3Exam"),
]

def fetch_title(aid):
    url = f"https://arxiv.org/abs/{aid}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) research/1.0"})
    raw = urllib.request.urlopen(req, timeout=40).read().decode("utf-8", "ignore")
    m = re.search(r'<meta name="citation_title" content="([^"]*)"', raw)
    if m:
        return m.group(1).strip()
    m2 = re.search(r"<title>(.*?)</title>", raw, re.S)
    return re.sub(r"\s+", " ", m2.group(1)).strip() if m2 else ""

results = {}
for aid, expect in CANDIDATES:
    for attempt in range(3):
        try:
            title = fetch_title(aid)
            ok = expect.lower() in title.lower()
            results[aid] = {"ok": ok, "title": title}
            print(f"[{'✓' if ok else '✗'}] {aid}: {title[:90]}")
            break
        except Exception as e:
            if attempt == 2:
                results[aid] = {"ok": False, "title": f"ERROR {e}"}
                print(f"[✗] {aid} 失败: {e}")
            else:
                time.sleep(6 * (attempt + 1))
    time.sleep(2)

json.dump(results, open("fullEnglish/refs_verified.json", "w"), ensure_ascii=False, indent=2)
ok_n = sum(1 for v in results.values() if v["ok"])
print(f"\n核实结果: {ok_n}/{len(CANDIDATES)} 个引用确认真实存在")
