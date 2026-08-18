# 英文全科"弱教师"学生超越（Qwen3-32B 弱教师 → Qwen2.5-32B / Llama-70B）

## 实验目标

在英文全科（MedQA+MedMCQA+MMLU 共 8293 题）上验证"弱教师"超越：用一个零样本与学生接近的开源模型（Qwen3-32B）当"教师"标杆，证明更弱的开源学生（Qwen2.5-32B、Llama-70B）经蒸馏后反超它——即"老模型经蒸馏后反超新模型"。

核心机制：弱教师零样本领先学生(headroom)仅 1.41~2.46pp，小于蒸馏增益(~4pp)，故训练后反超。

## 历史结果

| 学生 | 学生零样本 | 学生训练后 | 弱教师 Qwen3-32B(73.84%) | 超越幅度 |
|---|---|---|---|---|
| Qwen2.5-32B | 71.38% | 75.64% | 领先 2.46pp | **+1.80pp ✓** |
| Llama-70B | 72.43% | 76.27% | 领先 1.41pp | **+2.43pp ✓** |

（对比：强教师 flash 79.80%，四学生训练后 72.41/75.64/76.27/77.60 全部追不上——弱教师能超、强教师超不了，是 headroom 相变的证据。）

## 目录内容

- `data/test_medqa.jsonl`（1273）、`test_medmcqa.jsonl`（4183）、`test_mmlu.jsonl`（2837）、`test_pubmedqa.jsonl`（1000）
- `scripts/eval_full_ladder.py`：零样本阶梯（4 个模型在英文全科 8293 题的零样本）
- `scripts/eval_weakteacher_full.py`：弱教师组合评估（训练后学生 vs 弱教师，全科 8293 题）

## 依赖（重要）

本文件夹是**对 fullEnglish 主实验已训练 adapter 的再分析**，不重复训练：

- 学生训练后 adapter 复用 `fullEnglish/03_main_distill/runs/32B_a00_*`（Qwen2.5-32B）与 `Llama70B_a00_*`（Llama-70B，QLoRA 4bit）的 `best/`。
- 基座模型：`../models/Qwen3-32B`、`../models/Qwen2.5-32B-Instruct`、`../models/Qwen2.5-14B-Instruct`、`../models/Llama-3.3-70B-Instruct`。
- 训练代码：`../shared/train_choice_head_distill.py`。

## 执行步骤

```bash
cd 24_english_general_surpass

# 1. 零样本阶梯（4 模型在英文全科 8293 题零样本，得到弱教师 Qwen3-32B=73.84%）
python scripts/eval_full_ladder.py

# 2. 弱教师组合评估（训练后学生 vs 弱教师）
python scripts/eval_weakteacher_full.py
```

预期输出（与历史结果一致）：

```
弱教师 Qwen3-32B: 73.84%
学生 Qwen2.5-32B 训练后: 75.64% → 超 +1.80pp
学生 Llama-70B 训练后: 76.27% → 超 +2.43pp
```

## 关键结论

弱教师（Qwen3-32B）领先学生 <4pp → 学生训练后全部反超；强教师（flash 79.80%）领先 6~12pp → 学生全部追不上（四学生单调线 7.39→4.16→3.53→2.21 始终未过零）。同一规律正反两向成立。
