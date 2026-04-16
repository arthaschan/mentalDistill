#!/usr/bin/env python3
"""
train_adaptive_alpha_distill.py — 样本自适应 α-散度蒸馏

信息几何原理：
  不同 α 值在 Fisher 流形上具有不同的几何含义：
    α=1  (KL前向): 重视教师高概率区域, 对错误标签梯度大
    α=0  (Hellinger): 对所有概率区域平等对待, 梯度稳定
    α=-1 (KL反向): 重视学生高概率区域, mode-covering

  F22 发现 KL 种子敏感性最大(Δ=4.82pp), Hellinger 最稳健。
  F26 发现边界距离可区分教师自信/犹豫样本。

策略：
  根据每个样本的 Fisher 边界距离 d_b 动态选择 α：
    α_i = sigmoid((d_b - tau) / gamma) * (alpha_high - alpha_low) + alpha_low
  
  教师自信(d_b 小) → α 接近 alpha_low(如 1.0, 用 KL 精确学习)
  教师犹豫(d_b 大) → α 接近 alpha_high(如 0.0, 用 Hellinger 稳健学习)

基于 shared/train_alpha_distill.py 修改。
"""
import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup


OPTION_LETTERS = ["A", "B", "C", "D", "E"]


def set_global_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def extract_answer_char(text: str) -> str:
    for ch in text.strip().upper():
        if ch in OPTION_LETTERS:
            return ch
    return ""


def boundary_distance(probs, eps=1e-12):
    """Fisher-Rao boundary distance: 2 * arcsin(sqrt(p_min))"""
    p_min = min(max(p, eps) for p in probs)
    return 2.0 * math.asin(math.sqrt(p_min))


def adaptive_alpha(d_b: float, tau: float, gamma: float,
                   alpha_low: float, alpha_high: float) -> float:
    """Map boundary distance to α via sigmoid.
    
    Small d_b (confident teacher) → α close to alpha_low (e.g. 1.0 = KL)
    Large d_b (uncertain teacher) → α close to alpha_high (e.g. 0.0 = Hellinger)
    """
    sig = 1.0 / (1.0 + math.exp(-(d_b - tau) / max(gamma, 1e-8)))
    return alpha_low + sig * (alpha_high - alpha_low)


def evaluate_generation(model, tokenizer, file_path, device, max_new_tokens=4):
    samples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            q = row.get("Question", "")
            opts = row.get("Options", "")
            ans = row.get("Answer", "")
            if q and opts and ans:
                samples.append((q, opts, ans))

    correct = 0
    model.eval()
    for q, opts, ans in samples:
        prompt = (
            "<|im_start|>system\n"
            "你是一名专业的牙科医生，只需输出一个字母（A、B、C、D、E）作为结果，不要附带任何解释或空格。\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"问题：{q}\n选项：\n{opts}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
        gen = tokenizer.decode(outputs[0][inputs["input_ids"].size(1):], skip_special_tokens=True)
        pred = extract_answer_char(gen)
        if pred == ans:
            correct += 1
    return 100.0 * correct / len(samples) if samples else 0.0


class AdaptiveAlphaDataset(Dataset):
    """数据集：为每个样本预计算 per-sample α 值。"""
    def __init__(self, data_path: str, tokenizer, max_length: int = 1024,
                 tau: float = 0.05, gamma: float = 0.02,
                 alpha_low: float = 1.0, alpha_high: float = 0.0,
                 default_distill_mask: int = 1):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.default_distill_mask = 1 if int(default_distill_mask) != 0 else 0
        self.data = []
        alpha_values = []

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.data.append(json.loads(line))

        for row in self.data:
            dist = row.get("TeacherDist", None)
            if not isinstance(dist, dict):
                row["_per_alpha"] = (alpha_low + alpha_high) / 2  # default
                continue

            probs = []
            for ch in OPTION_LETTERS:
                try:
                    probs.append(max(0.0, float(dist.get(ch, 0.0))))
                except Exception:
                    probs.append(0.0)
            s = sum(probs)
            if s <= 0:
                row["_per_alpha"] = (alpha_low + alpha_high) / 2
                continue

            probs = [p / s for p in probs]
            d_b = boundary_distance(probs)
            a = adaptive_alpha(d_b, tau, gamma, alpha_low, alpha_high)
            row["_per_alpha"] = a
            alpha_values.append(a)

        if alpha_values:
            print(f"[DATA] loaded={len(self.data)} from {data_path}")
            print(f"[ADAPTIVE-α] tau={tau}, gamma={gamma}, α_low={alpha_low}, α_high={alpha_high}")
            print(f"  α stats: mean={np.mean(alpha_values):.4f}, std={np.std(alpha_values):.4f}, "
                  f"min={min(alpha_values):.4f}, max={max(alpha_values):.4f}")
            # Distribution histogram
            bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.01]
            hist = np.histogram(alpha_values, bins=bins)[0]
            print(f"  α distribution: " + " | ".join(f"[{bins[i]:.1f},{bins[i+1]:.1f}):{hist[i]}" for i in range(len(hist))))

    def __len__(self):
        return len(self.data)

    def _build_teacher_dist(self, row, gt_letter: str):
        raw = row.get("TeacherDist", None)
        probs = torch.zeros(len(OPTION_LETTERS), dtype=torch.float32)
        if isinstance(raw, dict):
            for i, ch in enumerate(OPTION_LETTERS):
                try:
                    probs[i] = float(raw.get(ch, 0.0))
                except Exception:
                    probs[i] = 0.0
        if float(probs.sum().item()) <= 0.0 and gt_letter in OPTION_LETTERS:
            probs[OPTION_LETTERS.index(gt_letter)] = 1.0
        probs = torch.clamp(probs, min=0.0)
        s = float(probs.sum().item())
        if s <= 0.0:
            probs[0] = 1.0
            s = 1.0
        return probs / s

    def __getitem__(self, idx):
        row = self.data[idx]
        q = str(row.get("Question", ""))
        opts = row.get("Options", "")
        ans = str(row.get("Answer", "")).strip().upper()
        if ans not in OPTION_LETTERS:
            ans = "A"

        prompt_prefix = (
            "<|im_start|>system\n"
            "你是一名专业的牙科医生，只需输出一个字母（A、B、C、D、E）作为结果，不要附带任何解释或空格。\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"问题：{q}\n选项：\n{opts}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        text = prompt_prefix + f"{ans}<|im_end|>"
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        prefix_enc = self.tokenizer(prompt_prefix, truncation=True, max_length=self.max_length)
        prefix_len = len(prefix_enc["input_ids"])

        labels = enc["input_ids"].squeeze().clone()
        labels[:prefix_len] = -100
        labels[enc["attention_mask"].squeeze() == 0] = -100

        distill_mask = 1 if str(row.get("SelectiveSource", "")).strip() == "clean_teacher" else self.default_distill_mask

        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": labels,
            "teacher_dist": self._build_teacher_dist(row, ans),
            "gt_option": torch.tensor(OPTION_LETTERS.index(ans), dtype=torch.long),
            "distill_mask": torch.tensor(distill_mask, dtype=torch.float32),
            "per_alpha": torch.tensor(row["_per_alpha"], dtype=torch.float32),
        }


# ─── α-divergence 核心实现 ─────────────────────────────────────────
def alpha_divergence(student_logits_5: torch.Tensor, teacher_dist_5: torch.Tensor,
                     div_alpha: float, eps: float = 1e-8) -> torch.Tensor:
    q = F.softmax(student_logits_5, dim=-1)
    p = teacher_dist_5.clamp(min=eps)

    if abs(div_alpha - 1.0) < 1e-6:
        return (p * (p.log() - q.clamp(min=eps).log())).sum()
    elif abs(div_alpha + 1.0) < 1e-6:
        return (q * (q.clamp(min=eps).log() - p.log())).sum()
    elif abs(div_alpha) < 1e-6:
        bc = (p.sqrt() * q.clamp(min=eps).sqrt()).sum()
        return 2.0 * (1.0 - bc)
    else:
        a = div_alpha
        exp_p = (1.0 + a) / 2.0
        exp_q = (1.0 - a) / 2.0
        integral = (p.pow(exp_p) * q.clamp(min=eps).pow(exp_q)).sum()
        coeff = 4.0 / (1.0 - a * a)
        return coeff * (1.0 - integral)


def adaptive_alpha_distill_loss(student_logits, labels, teacher_dist, gt_option,
                                option_token_ids, ce_weight, distill_mask, per_alpha):
    """Per-sample adaptive α-divergence loss.
    
    Each sample uses its own div_alpha from per_alpha tensor.
    ce_weight controls the CE vs divergence balance (same for all samples).
    """
    shift_logits = student_logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    valid_positions = shift_labels != -100

    ce_terms = []
    div_terms = []
    ce_ws = []
    div_ws = []

    for b in range(shift_logits.size(0)):
        pos_idx = torch.nonzero(valid_positions[b], as_tuple=False)
        if pos_idx.numel() == 0:
            continue
        pos = int(pos_idx[0].item())
        opt_logits = shift_logits[b, pos, option_token_ids]

        ce_b = F.cross_entropy(opt_logits.view(1, -1), gt_option[b].view(1), reduction="sum")
        ce_terms.append(ce_b)
        ce_ws.append(1.0)

        tdist = teacher_dist[b].to(student_logits.device).float()
        tdist = torch.clamp(tdist, min=0.0)
        tsum = float(tdist.sum().item())
        if tsum > 0:
            tdist = tdist / tsum
            sample_alpha = float(per_alpha[b].item())
            div_b = alpha_divergence(opt_logits, tdist, sample_alpha)
            w = float(distill_mask[b].item())
            div_terms.append(div_b * w)
            div_ws.append(w)

    if ce_terms:
        ce_loss = torch.stack(ce_terms).sum() / max(1.0, float(sum(ce_ws)))
    else:
        ce_loss = torch.zeros((), device=student_logits.device, dtype=student_logits.dtype)

    if div_terms and sum(div_ws) > 0:
        div_loss = torch.stack(div_terms).sum() / max(1e-8, float(sum(div_ws)))
    else:
        div_loss = torch.zeros((), device=student_logits.device, dtype=student_logits.dtype)

    kl_weight = 1.0 - ce_weight
    return kl_weight * div_loss + ce_weight * ce_loss


def main():
    parser = argparse.ArgumentParser(description="Adaptive α-divergence distillation")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, default="")
    parser.add_argument("--test_path", type=str, default="")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1.2e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.35,
                        help="CE weight: loss = (1-alpha)*div + alpha*CE (kept for compatibility)")
    parser.add_argument("--tau", type=float, default=0.05,
                        help="Boundary distance center for sigmoid mapping")
    parser.add_argument("--gamma", type=float, default=0.02,
                        help="Sigmoid temperature for α mapping")
    parser.add_argument("--alpha_low", type=float, default=1.0,
                        help="α value for confident samples (small d_b)")
    parser.add_argument("--alpha_high", type=float, default=0.0,
                        help="α value for uncertain samples (large d_b)")
    parser.add_argument("--default_distill_mask", type=int, choices=[0, 1], default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    set_global_seed(args.seed, args.deterministic)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
        unk_token="<|endoftext|>",
        trust_remote_code=True,
    )

    option_token_ids = []
    for ch in OPTION_LETTERS:
        tids = tokenizer.encode(ch, add_special_tokens=False)
        option_token_ids.append(tids[0])
    option_token_ids = torch.tensor(option_token_ids, dtype=torch.long, device=device)

    ds = AdaptiveAlphaDataset(
        args.data_path, tokenizer,
        tau=args.tau, gamma=args.gamma,
        alpha_low=args.alpha_low, alpha_high=args.alpha_high,
        default_distill_mask=args.default_distill_mask,
    )
    g = torch.Generator()
    g.manual_seed(args.seed)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, generator=g)

    base = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, trust_remote_code=True)
    lora_cfg = LoraConfig(
        task_type="CAUSAL_LM",
        inference_mode=False,
        r=args.rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(base, lora_cfg)
    model = model.to(device)
    model.train()

    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    accum_steps = max(1, args.gradient_accumulation_steps)

    best_val_acc = -1.0
    best_val_epoch = -1

    for ep in range(args.num_epochs):
        pbar = tqdm(dl, desc=f"AdaptiveAlpha Epoch {ep + 1}/{args.num_epochs}")
        optim.zero_grad(set_to_none=True)
        for i, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            tdist = batch["teacher_dist"].to(device)
            gt_opt = batch["gt_option"].to(device)
            dmask = batch["distill_mask"].to(device)
            p_alpha = batch["per_alpha"].to(device)

            out = model(input_ids=input_ids, attention_mask=attn)
            loss = adaptive_alpha_distill_loss(
                out.logits, labels, tdist, gt_opt,
                option_token_ids, ce_weight=args.alpha,
                distill_mask=dmask, per_alpha=p_alpha,
            )
            loss = loss / accum_steps
            loss.backward()

            if ((i + 1) % accum_steps == 0) or (i + 1 == len(dl)):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                optim.zero_grad(set_to_none=True)

            pbar.set_postfix(loss=f"{float(loss.item() * accum_steps):.4f}")

        ckpt_dir = os.path.join(args.output_dir, "checkpoints", f"epoch_{ep + 1}")
        os.makedirs(ckpt_dir, exist_ok=True)
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)

        if args.val_path:
            val_acc = evaluate_generation(model, tokenizer, args.val_path, device)
            print(f"[VAL] epoch={ep + 1} acc={val_acc:.2f}%")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_epoch = ep + 1
                best_dir = os.path.join(args.output_dir, "best")
                os.makedirs(best_dir, exist_ok=True)
                model.save_pretrained(best_dir)
                tokenizer.save_pretrained(best_dir)

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.test_path and best_val_epoch > 0:
        print(f"[BEST] val_acc={best_val_acc:.2f}% at epoch {best_val_epoch}")
        best_dir = os.path.join(args.output_dir, "best")
        if os.path.isdir(best_dir):
            best_base = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, trust_remote_code=True)
            best_model = PeftModel.from_pretrained(best_base, best_dir)
            best_model = best_model.to(device)
            test_acc = evaluate_generation(best_model, tokenizer, args.test_path, device)
            print(f"[TEST-BEST] epoch={best_val_epoch} test_acc={test_acc:.2f}%")
            del best_model, best_base
            torch.cuda.empty_cache()
    elif args.test_path:
        test_acc = evaluate_generation(model, tokenizer, args.test_path, device)
        print(f"测试集准确率: {test_acc:.2f}%")


if __name__ == "__main__":
    main()
