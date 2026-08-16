#!/usr/bin/env python3
"""从 HF 下载 Qwen/Qwen3-32B（非 gated，用 hf-mirror 加速）到本地，供零样本测 headroom。"""
import os
from huggingface_hub import snapshot_download

repo = "Qwen/Qwen3-32B"
local = "models/Qwen3-32B"
print(f"开始下载 {repo} -> {local}", flush=True)
path = snapshot_download(repo_id=repo, local_dir=local)
print(f"下载完成: {path}", flush=True)
