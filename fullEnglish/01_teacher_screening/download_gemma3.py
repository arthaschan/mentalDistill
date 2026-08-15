#!/usr/bin/env python3
"""从 ModelScope 下载 gemma-3-27b-it（不设门槛）到本地，供 B 任务 headroom 测量。"""
import sys
from modelscope import snapshot_download

model_id = "google/gemma-3-27b-it"
local_dir = "models/gemma-3-27b-it"

print(f"开始下载 {model_id} -> {local_dir}", flush=True)
path = snapshot_download(model_id, local_dir=local_dir)
print(f"下载完成: {path}", flush=True)
