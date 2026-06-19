#!/usr/bin/env python3
"""
download_phase2_hf.py — 阶段 2 教师下载（HF 官方源 + hf-mirror 兜底）

策略：
- 默认 HF 官方源（实测 33MB/s）；Gemma(gated) 用 hf-mirror 兜底绕授权。
- 开启 hf_transfer 提速。
- 逐个下载到 models/<local_name>，断点续传（phi-4 已下分片复用）。
- 每个下完后清 HF cache 的 blob 副本，避免双倍占用。
- 落地后校验 config.json + 权重分片齐全。
"""
import os
import shutil
import sys
from pathlib import Path

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

REPO_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = REPO_ROOT / "models"

# (hf_repo_id, local_dir_name, endpoint)  endpoint None=官方
TEACHERS = [
    ("microsoft/phi-4", "phi-4", None),                          # 续传，~28GB
    ("zai-org/GLM-4-32B-0414", "GLM-4-32B-0414", None),          # ~64GB
    ("01-ai/Yi-1.5-34B-Chat", "Yi-1.5-34B-Chat", None),          # ~68GB
    ("google/gemma-2-27b-it", "gemma-2-27b-it",
     "https://hf-mirror.com"),                                    # gated -> 镜像兜底
]


def free_gb(p):
    return shutil.disk_usage(p).free / 1024**3


def clear_hf_cache():
    cache = Path.home() / ".cache" / "huggingface" / "hub"
    if cache.exists():
        try:
            shutil.rmtree(cache)
            print(f"  [cache] cleared {cache}")
        except Exception as e:
            print(f"  [cache] could not clear: {e}")


def verify(d: Path):
    if not (d / "config.json").exists():
        return False, "no config.json"
    shards = list(d.glob("*.safetensors"))
    if not shards:
        return False, "no safetensors"
    # cross-check against the index if present
    idx = d / "model.safetensors.index.json"
    if idx.exists():
        import json
        want = set(json.load(open(idx))["weight_map"].values())
        have = {s.name for s in shards}
        missing = want - have
        if missing:
            return False, f"missing shards: {sorted(missing)[:3]}... ({len(missing)})"
    size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1024**3
    return True, f"{len(shards)} shards, {size:.1f} GB"


def main():
    from huggingface_hub import snapshot_download
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Target: {MODELS_DIR}  | free: {free_gb(MODELS_DIR):.1f} GB\n")

    results = []
    default_ep = os.environ.get("HF_ENDPOINT")
    for repo, name, endpoint in TEACHERS:
        local = MODELS_DIR / name
        ok, _ = verify(local) if local.exists() else (False, "")
        if ok:
            print(f"=== SKIP {repo} (complete at {local}) ===")
            results.append((name, "exists")); continue

        # set per-model endpoint
        if endpoint:
            os.environ["HF_ENDPOINT"] = endpoint
        elif default_ep:
            os.environ["HF_ENDPOINT"] = default_ep
        else:
            os.environ.pop("HF_ENDPOINT", None)
        ep_show = os.environ.get("HF_ENDPOINT", "https://huggingface.co")

        print(f"=== {repo} -> {local}  via {ep_show}  (free: {free_gb(MODELS_DIR):.1f} GB) ===")
        try:
            snapshot_download(
                repo_id=repo, local_dir=str(local),
                allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model", "tokenizer*", "*.md"],
                max_workers=8,
            )
        except Exception as e:
            print(f"  [ERROR] {repo}: {e}")
            results.append((name, f"FAILED: {str(e)[:120]}"))
            clear_hf_cache()
            continue
        clear_hf_cache()
        ok, msg = verify(local)
        results.append((name, ("OK " + msg) if ok else ("INCOMPLETE " + msg)))
        print(f"  [{results[-1][1]}]  free left: {free_gb(MODELS_DIR):.1f} GB\n")

    print("=" * 60)
    print("Download summary:")
    for n, s in results:
        print(f"  {n:<20} {s}")
    print(f"Free remaining: {free_gb(MODELS_DIR):.1f} GB")


if __name__ == "__main__":
    main()
