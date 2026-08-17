#!/usr/bin/env python3
"""搜索 DentalQA/DentalBench 数据集的实际托管位置。"""
import json, urllib.request, urllib.parse

def get(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 research"})
    return urllib.request.urlopen(req, timeout=timeout).read().decode("utf-8", "ignore")

# 1) HF title search，多关键词 + 更多结果
print("========== HuggingFace ==========")
for q in ["dental", "DentalQA", "DentalBench", "stomatology", "dentistry", "oral medicine"]:
    url = "https://huggingface.co/api/datasets?search=" + urllib.parse.quote(q) + "&limit=50"
    try:
        data = json.loads(get(url))
        hits = [d for d in data if any(k in (d.get("id","")+d.get("description","")+str(d.get("tags",""))).lower()
                for k in ["dental","dentist","stomat","oral","口腔"])]
        print(f"--- {q} (total {len(data)}) ---")
        for d in hits[:15]:
            print(f"  {d['id']:45s} dl={d.get('downloads',0):>6}  {(d.get('description') or '')[:60]}")
    except Exception as e:
        print(f"  [{q}] err {e}")

# 2) ModelScope
print("\n========== ModelScope ==========")
for q in ["DentalBench", "DentalQA", "口腔医学", "牙科"]:
    url = "https://modelscope.cn/api/v1/dolphin/datasets?query=" + urllib.parse.quote(q) + "&page_size=20"
    try:
        raw = get(url)
        data = json.loads(raw)
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = data.get("Data", {}).get("Datasets", []) if isinstance(data.get("Data"), dict) else data.get("Data", [])
        else:
            items = []
        print(f"--- {q} ({len(items)} hits) ---")
        for x in items[:10]:
            if isinstance(x, dict):
                print(f"  {x.get('Name') or x.get('Path') or x.get('Id')}  {(x.get('Description') or '')[:70]}")
    except Exception as e:
        print(f"  [{q}] err {e}")
