#!/usr/bin/env python3
"""Download Gemma-2-27b-it via HF official endpoint with token (gated model)."""
import os, shutil
from pathlib import Path
os.environ["HF_HUB_ENABLE_HF_TRANSFER"]="1"
os.environ.pop("HF_ENDPOINT", None)  # force official hf.co
REPO_ROOT=Path(__file__).resolve().parents[3]
local=REPO_ROOT/"models"/"gemma-2-27b-it"
from huggingface_hub import snapshot_download
tok=os.environ.get("HF_TOKEN")
print("token present:", bool(tok), "| target:", local)
snapshot_download(repo_id="google/gemma-2-27b-it", local_dir=str(local),
    token=tok,
    allow_patterns=["*.safetensors","*.json","*.txt","*.model","tokenizer*","*.md"],
    max_workers=8)
# clear cache
cache=Path.home()/".cache"/"huggingface"/"hub"
if cache.exists(): shutil.rmtree(cache, ignore_errors=True)
# verify
import glob, json
shards=glob.glob(str(local/"*.safetensors"))
idx=local/"model.safetensors.index.json"
ok="config.json exists" if (local/"config.json").exists() else "NO config"
print(f"done: {len(shards)} shards, {ok}")
