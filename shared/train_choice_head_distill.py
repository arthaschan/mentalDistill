import argparse
import json
import os
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_cosine_schedule_with_warmup


OPTION_LETTERS = ["A", "B", "C", "D", "E"]


# Prompt 语言可切换: 默认 zh (原中文牙科 prompt, 保持 CMExam 行为完全不变);
# 设 DISTILL_PROMPT_LANG=en 用英文通用医学 prompt (跨数据集实验如 MedQA 用)。
_PROMPT_LANG = os.environ.get("DISTILL_PROMPT_LANG", "zh").lower()

# 设 DISTILL_USE_CHAT_TEMPLATE=1 时用 tokenizer 自带 chat template 拼 prompt
# （Llama/Gemma 等非 Qwen 模型用；默认 0 保持 Qwen 硬编码 <|im_start|> 历史行为不变）。
_USE_CHAT_TEMPLATE = os.environ.get("DISTILL_USE_CHAT_TEMPLATE", "0") == "1"


def build_mcq_prompt(q, opts):
    """返回 (system_line, user_block) 用于拼 chat prompt。"""
    if _PROMPT_LANG == "en":
        system_line = ("You are a medical expert. Output exactly one letter "
                       "(A, B, C, D, or E) as the answer, with no explanation or spaces.\n")
        user_block = f"Question: {q}\nOptions:\n{opts}\n"
    else:
        system_line = ("你是一名专业的牙科医生，只需输出一个字母（A、B、C、D、E）作为结果，不要附带任何解释或空格。\n")
        user_block = f"问题：{q}\n选项：\n{opts}\n"
    return system_line, user_block


def apply_prompt_template(tokenizer, sys_line, user_block):
    """把 system/user 拼成 (prompt_prefix, 答案后闭合串)。

    _USE_CHAT_TEMPLATE=1 时用 tokenizer 自带 chat template（Llama/Gemma 等），
    否则用 Qwen 硬编码 <|im_start|> 格式（默认，保持历史行为不变）。
    """
    if _USE_CHAT_TEMPLATE:
        msgs = [
            {"role": "system", "content": sys_line},
            {"role": "user", "content": user_block},
        ]
        prefix = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        return prefix, ""
    prefix = (
        "<|im_start|>system\n"
        + sys_line
        + "<|im_end|>\n"
        "<|im_start|>user\n"
        + user_block
        + "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return prefix, "<|im_end|>"


def load_base_model(model_name, quantize, device):
    """加载 base 模型。quantize='4bit' 走 QLoRA(bitsandbytes NF4)，否则 bf16。"""
    if quantize == "4bit":
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        return AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_cfg,
            device_map={"": device}, trust_remote_code=True,
        )
    return AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
    )


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
        sys_line, user_block = build_mcq_prompt(q, opts)
        prompt, _ = apply_prompt_template(tokenizer, sys_line, user_block)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
        gen = tokenizer.decode(outputs[0][inputs["input_ids"].size(1):], skip_special_tokens=True)
        pred = extract_answer_char(gen)
        if pred == ans:
            correct += 1
    return 100.0 * correct / len(samples) if samples else 0.0


class DentalChoiceHeadDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 1024, default_distill_mask: int = 1):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.default_distill_mask = 1 if int(default_distill_mask) != 0 else 0
        self.data = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.data.append(json.loads(line))
        print(f"[DATA] loaded={len(self.data)} from {data_path}")

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

        sys_line, user_block = build_mcq_prompt(q, opts)
        prompt_prefix, closing = apply_prompt_template(self.tokenizer, sys_line, user_block)
        text = prompt_prefix + f"{ans}{closing}"
        # 不 padding（动态 padding 由 collate_fn 按 batch 内最长处理），避免把中位数 272 的
        # 序列强行补到 1024 浪费 ~4 倍算力。loss 只看答案 token，padding 不影响结果。
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        prefix_enc = self.tokenizer(prompt_prefix, truncation=True, max_length=self.max_length)
        prefix_len = len(prefix_enc["input_ids"])

        labels = enc["input_ids"].squeeze().clone()
        labels[:prefix_len] = -100

        distill_mask = 1 if str(row.get("SelectiveSource", "")).strip() == "clean_teacher" else self.default_distill_mask

        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": labels,
            "teacher_dist": self._build_teacher_dist(row, ans),
            "gt_option": torch.tensor(OPTION_LETTERS.index(ans), dtype=torch.long),
            "distill_mask": torch.tensor(distill_mask, dtype=torch.float32),
        }


def choice_head_distill_loss(student_logits, labels, teacher_dist, gt_option, option_token_ids, alpha, distill_mask):
    shift_logits = student_logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    valid_positions = shift_labels != -100

    ce_terms = []
    kl_terms = []
    ce_weights = []
    kl_weights = []

    for b in range(shift_logits.size(0)):
        pos_idx = torch.nonzero(valid_positions[b], as_tuple=False)
        if pos_idx.numel() == 0:
            continue
        pos = int(pos_idx[0].item())
        opt_logits = shift_logits[b, pos, option_token_ids]

        ce_b = F.cross_entropy(opt_logits.view(1, -1), gt_option[b].view(1), reduction="sum")
        ce_terms.append(ce_b)
        ce_weights.append(1.0)

        tdist = teacher_dist[b].to(student_logits.device).float()
        tdist = torch.clamp(tdist, min=0.0)
        tsum = float(tdist.sum().item())
        if tsum > 0:
            tdist = tdist / tsum
            slogp = F.log_softmax(opt_logits, dim=-1)
            kl_b = F.kl_div(slogp, tdist, reduction="sum")
            w = float(distill_mask[b].item())
            kl_terms.append(kl_b * w)
            kl_weights.append(w)

    if ce_terms:
        ce_loss = torch.stack(ce_terms).sum() / max(1.0, float(sum(ce_weights)))
    else:
        ce_loss = torch.zeros((), device=student_logits.device, dtype=student_logits.dtype)

    if kl_terms and sum(kl_weights) > 0:
        kl_loss = torch.stack(kl_terms).sum() / float(sum(kl_weights))
    else:
        kl_loss = torch.zeros((), device=student_logits.device, dtype=student_logits.dtype)

    return alpha * kl_loss + (1.0 - alpha) * ce_loss


def main():
    parser = argparse.ArgumentParser(description="Choice-head distillation for dental MCQ")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, default="")
    parser.add_argument("--test_path", type=str, default="")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=1024,
                        help="序列截断上限（动态 padding，只做截断不补 1024）")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1.2e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument("--default_distill_mask", type=int, choices=[0, 1], default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--quantize", type=str, choices=["none", "4bit"], default="none",
                        help="模型加载量化: none=bf16(默认), 4bit=QLoRA(bitsandbytes NF4, 供 70B 等大模型)")
    parser.add_argument("--resume_from", type=str, default="")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Fraction of total steps for linear warmup (0=no warmup)")
    parser.add_argument("--use_cosine_schedule", action="store_true",
                        help="Use cosine LR schedule with warmup")
    args = parser.parse_args()

    set_global_seed(args.seed, args.deterministic)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if _USE_CHAT_TEMPLATE:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    else:
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

    ds = DentalChoiceHeadDataset(
        args.data_path,
        tokenizer,
        max_length=args.max_length,
        default_distill_mask=args.default_distill_mask,
    )

    def collate_fn(batch):
        """动态 padding：按 batch 内最长序列补齐，避免固定 1024 浪费算力。"""
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        max_len = max(b["input_ids"].size(0) for b in batch)

        def _pad(t, val):
            return F.pad(t, (0, max_len - t.size(0)), value=val)

        return {
            "input_ids": torch.stack([_pad(b["input_ids"], pad_id) for b in batch]),
            "attention_mask": torch.stack([_pad(b["attention_mask"], 0) for b in batch]),
            "labels": torch.stack([_pad(b["labels"], -100) for b in batch]),
            "teacher_dist": torch.stack([b["teacher_dist"] for b in batch]),
            "gt_option": torch.stack([b["gt_option"] for b in batch]),
            "distill_mask": torch.stack([b["distill_mask"] for b in batch]),
        }

    g = torch.Generator()
    g.manual_seed(args.seed)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, generator=g, collate_fn=collate_fn)

    base = load_base_model(args.model_name, args.quantize, device)
    if args.resume_from and os.path.isdir(args.resume_from):
        model = PeftModel.from_pretrained(base, args.resume_from, is_trainable=True)
    else:
        lora_cfg = LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            r=args.rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        model = get_peft_model(base, lora_cfg)

    # 4bit 模型已由 device_map 放置，.to(device) 多余且可能破坏量化层；bf16 才需要显式搬 GPU。
    if args.quantize != "4bit":
        model = model.to(device)
    model.train()

    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    accum_steps = max(1, args.gradient_accumulation_steps)
    total_optim_steps = (len(dl) + accum_steps - 1) // accum_steps * args.num_epochs
    warmup_steps = int(total_optim_steps * args.warmup_ratio) if args.use_cosine_schedule else 0
    scheduler = None
    if args.use_cosine_schedule:
        scheduler = get_cosine_schedule_with_warmup(optim, num_warmup_steps=warmup_steps, num_training_steps=total_optim_steps)
        print(f"[SCHED] cosine schedule: {total_optim_steps} total steps, {warmup_steps} warmup steps")

    best_val_acc = -1.0
    best_val_epoch = -1

    global_step = 0
    for ep in range(args.num_epochs):
        pbar = tqdm(dl, desc=f"HeadDistill Epoch {ep + 1}/{args.num_epochs}")
        optim.zero_grad(set_to_none=True)
        for i, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            tdist = batch["teacher_dist"].to(device)
            gt_opt = batch["gt_option"].to(device)
            dmask = batch["distill_mask"].to(device)

            out = model(input_ids=input_ids, attention_mask=attn)
            loss = choice_head_distill_loss(
                out.logits,
                labels,
                tdist,
                gt_opt,
                option_token_ids,
                alpha=args.alpha,
                distill_mask=dmask,
            )
            loss = loss / accum_steps
            loss.backward()

            if ((i + 1) % accum_steps == 0) or (i + 1 == len(dl)):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                if scheduler is not None:
                    scheduler.step()
                optim.zero_grad(set_to_none=True)
                global_step += 1

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

    # Evaluate test set at best val epoch checkpoint
    if args.test_path and best_val_epoch > 0:
        print(f"[BEST] val_acc={best_val_acc:.2f}% at epoch {best_val_epoch}")
        best_dir = os.path.join(args.output_dir, "best")
        if os.path.isdir(best_dir):
            best_base = load_base_model(args.model_name, args.quantize, device)
            best_model = PeftModel.from_pretrained(best_base, best_dir)
            if args.quantize != "4bit":
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
