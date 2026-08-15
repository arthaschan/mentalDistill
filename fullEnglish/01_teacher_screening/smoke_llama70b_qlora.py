#!/usr/bin/env python3
"""Llama-3.3-70B 4bit 加载冒烟测试：验证权重完好 + chat template + QLoRA 前向。"""
import sys
import torch

sys.path.insert(0, "shared")
from train_choice_head_distill import load_base_model
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer

d = torch.device("cuda:0")

print("=== 1. tokenizer + chat template ===", flush=True)
tok = AutoTokenizer.from_pretrained("models/Llama-3.3-70B-Instruct", trust_remote_code=True)
print("tokenizer:", type(tok).__name__, "vocab", tok.vocab_size, flush=True)
msgs = [{"role": "system", "content": "You are a medical expert. Output exactly one letter."},
        {"role": "user", "content": "Question: test? Options: A. x B. y"}]
prefix = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
print("chat template 前缀:", repr(prefix[:90]), flush=True)

print("=== 2. 4bit 加载 (QLoRA) ===", flush=True)
m = load_base_model("models/Llama-3.3-70B-Instruct", "4bit", d)
print("4bit 加载 OK", flush=True)

cfg = LoraConfig(task_type="CAUSAL_LM", r=8, lora_alpha=16,
                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
m = get_peft_model(m, cfg)
print("LoRA 可训练参数:", sum(p.numel() for p in m.parameters() if p.requires_grad), flush=True)

print("=== 3. 前向 ===", flush=True)
inp = tok(prefix, return_tensors="pt").to(d)
with torch.no_grad():
    out = m(**inp)
print("前向 OK, logits shape", tuple(out.logits.shape), flush=True)
print("SMOKE_TEST_PASS", flush=True)
