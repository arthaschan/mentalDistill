#!/usr/bin/env python3
"""
download_phase2_teachers.py — 用 modelscope 国内源依次下载阶段 2 的 4 个新教师模型。

策略：逐个下载到 models/<local_name>；每个下完后清理 modelscope 缓存的 blob 副本，
避免缓存与目标目录双倍占用磁盘。落地后校验 config.json 与权重分片存在。
"""
import os
import shutil
import sys
import json
from pathlib import Path

# file is at research/distillability/scripts/, so repo root is parents[3]
REPO_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = REPO_ROOT / "models"

# (modelscope_repo_id, local_dir_name)
TEACHERS = [
    ("LLM-Research/phi-4", "phi-4"),                       # ~28GB, 最小先下
    ("LLM-Research/gemma-2-27b-it", "gemma-2-27b-it"),     # ~54GB
    ("ZhipuAI/GLM-4-32B-0414", "GLM-4-32B-0414"),          # ~64GB
    ("01ai/Yi-1.5-34B-Chat", "Yi-1.5-34B-Chat"),           # ~68GB
]


def free_gb(path):
    st = shutil.disk_usage(path)
    return st.free / (1024 ** 3)


def clear_modelscope_cache():
    """删除 modelscope 下载缓存的 blob 副本（已落地到 local_dir 后不再需要）。"""
    for cache in [Path.home() / ".cache" / "modelscope" / "hub",
                  Path.home() / ".cache" / "modelscope" / "models"]:
        if cache.exists():
            try:
                shutil.rmtree(cache)
                print(f"  [cache] cleared {cache}")
            except Exception as e:
                print(f"  [cache] could not clear {cache}: {e}")


def verify(local_dir: Path):
    cfg = local_dir / "config.json"
    if not cfg.exists():
        return False, "missing config.json"
    shards = list(local_dir.glob("*.safetensors")) + list(local_dir.glob("*.bin"))
    if not shards:
        return False, "no weight shards"
    return True, f"config.json + {len(shards)} weight files"


def main():
    from modelscope import snapshot_download
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Target: {MODELS_DIR}")
    print(f"Free now: {free_gb(MODELS_DIR):.1f} GB\n")

    results = []
    for repo, name in TEACHERS:
        local_dir = MODELS_DIR / name
        ok_existing, _ = verify(local_dir) if local_dir.exists() else (False, "")
        if ok_existing:
            print(f"=== SKIP {repo} (already present at {local_dir}) ===")
            results.append((name, "exists"))
            continue

        print(f"=== Downloading {repo} -> {local_dir}  (free: {free_gb(MODELS_DIR):.1f} GB) ===")
        try:
            snapshot_download(repo, local_dir=str(local_dir))
        except Exception as e:
            print(f"  [ERROR] download failed for {repo}: {e}")
            results.append((name, f"FAILED: {e}"))
            clear_modelscope_cache()
            continue

        clear_modelscope_cache()
        ok, msg = verify(local_dir)
        size_gb = sum(f.stat().st_size for f in local_dir.rglob("*") if f.is_file()) / (1024 ** 3)
        status = f"OK ({msg}, {size_gb:.1f} GB)" if ok else f"INCOMPLETE ({msg})"
        print(f"  [{status}]  free left: {free_gb(MODELS_DIR):.1f} GB\n")
        results.append((name, status))

    print("=" * 60)
    print("Download summary:")
    for name, status in results:
        print(f"  {name:<22} {status}")
    print(f"Free remaining: {free_gb(MODELS_DIR):.1f} GB")


if __name__ == "__main__":
    main()
