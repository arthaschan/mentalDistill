#!/usr/bin/env python3
"""并行下载 ModelScope 模型（多线程 + 断点续传），用于下载 gated 模型的镜像。

用法:
    python parallel_ms_download.py <model_id> <local_dir> [workers]

依赖 requests（已随 modelscope 安装）。
"""
import os
import sys
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

API = "https://modelscope.cn/api/v1/models/{mid}/repo/files?Revision=master&Root="
BASE = "https://modelscope.cn/models/{mid}/resolve/master/{name}"


def get_files(mid):
    r = requests.get(API.format(mid=mid), timeout=30)
    r.raise_for_status()
    data = r.json()
    files = data.get("Data", {}).get("Files", [])
    return [(f["Name"], int(f.get("Size", 0))) for f in files]


def download_one(mid, local_dir, name, size):
    dest = os.path.join(local_dir, name)
    part = dest + ".part"
    url = BASE.format(mid=mid, name=name)
    # 已完成
    if os.path.exists(dest) and os.path.getsize(dest) == size:
        return name, "skip"
    have = os.path.getsize(part) if os.path.exists(part) else 0
    if have > size:  # 损坏，重来
        os.remove(part)
        have = 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        if r.status_code not in (200, 206):
            r.raise_for_status()
        mode = "ab" if r.status_code == 206 else "wb"
        with open(part, mode) as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)
    if os.path.getsize(part) != size:
        raise IOError(f"{name} 大小不符: {os.path.getsize(part)} != {size}")
    os.rename(part, dest)
    return name, "done"


def main():
    mid = sys.argv[1]
    local_dir = sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    os.makedirs(local_dir, exist_ok=True)
    files = get_files(mid)
    total = sum(s for _, s in files)
    print(f"模型 {mid}: {len(files)} 个文件, 共 {total/1e9:.1f} GB, {workers} 线程", flush=True)

    done = 0
    lock = threading.Lock()
    t0 = time.time()

    def worker(fn):
        nonlocal done
        name, size = fn
        try:
            _, status = download_one(mid, local_dir, name, size)
        except Exception as e:
            return name, f"FAIL: {e}"
        with lock:
            done += size
            el = time.time() - t0
            rate = done / el / 1e6 if el > 0 else 0
            pct = done / total * 100
            print(f"  [{pct:5.1f}%] {done/1e9:.1f}/{total/1e9:.1f} GB "
                  f"{rate:.1f} MB/s  {status:6s} {name}", flush=True)
        return name, status

    results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(worker, fn): fn for fn in files}
        for fut in as_completed(futs):
            results.append(fut.result())

    fails = [r for r in results if r[1].startswith("FAIL")]
    print(f"\n完成: {len(results)-len(fails)}/{len(results)} 文件")
    if fails:
        for name, st in fails:
            print(f"  失败: {name} -> {st}")
        sys.exit(1)
    print(f"全部下载完成 -> {local_dir}")


if __name__ == "__main__":
    main()
