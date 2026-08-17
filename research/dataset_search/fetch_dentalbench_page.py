#!/usr/bin/env python3
import json, base64, re, urllib.request

# 拉 GitHub index.html 内容
req = urllib.request.Request(
    "https://api.github.com/repos/josa6f/DentalBench/contents/index.html",
    headers={"Accept": "application/vnd.github+json", "User-Agent": "research"},
)
d = json.load(urllib.request.urlopen(req, timeout=30))
if "content" not in d:
    print("ERR:", str(d)[:300])
    raise SystemExit(1)
html = base64.b64decode(d["content"]).decode("utf-8", "ignore")
open("research/dataset_search/dentalbench_index.html", "w").write(html)
print("index.html 字节:", len(html))

links = re.findall(r'(?:href|src|action)=["\']([^"\']+)["\']', html)
links = [l for l in links if not l.startswith("#") and not l.startswith("data:")]
print("链接(去重):")
for l in sorted(set(links)):
    print("  ", l)

text = re.sub(r"<[^>]+>", " ", html)
text = re.sub(r"\s+", " ", text)
for kw in ["download", "github", "huggingface", "zenodo", "drive", "DentalQA", "dataset", "hf.co"]:
    idx = 0
    found = False
    for m in re.finditer(kw, text, re.I):
        print(f"[{kw}] ...{text[max(0, m.start()-40):m.end()+80]}...")
        found = True
        break
