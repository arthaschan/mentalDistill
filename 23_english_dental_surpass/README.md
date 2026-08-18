# 英文牙科"弱教师"学生超越（Qwen3-32B 弱教师 → Qwen2.5-32B / Llama-70B）

## 实验目标

在英文牙科子集（980 题）上验证"弱教师"超越：用一个零样本与学生接近的开源模型（Qwen3-32B）当"教师"标杆，证明更弱的开源学生（Qwen2.5-32B、Llama-70B）经蒸馏后反超它——即"老模型经蒸馏后反超新模型"。

核心机制：弱教师零样本领先学生(headroom)仅 0.82~1.23pp，小于蒸馏增益(~4pp)，故训练后反超。

> 英文全科的"弱教师"超越见 `../24_english_general_surpass`。

## 历史结果

**英文牙科（980 题）**：

| 学生 | 学生零样本 | 学生训练后 | 弱教师 Qwen3-32B(62.76%) | 超越幅度 |
|---|---|---|---|---|
| Qwen2.5-32B | 61.53% | 65.65% | 领先 1.23pp | **+2.89pp ✓** |
| Llama-70B | 61.94% | 66.12%±2.18 | 领先 0.82pp | **+3.36pp ✓** |

（对比：强教师 flash 英文牙科 70.0%，学生 65.65%/66.12% 都追不上——弱教师能超、强教师超不了，正是 headroom 相变的证据。）

结果文件：`data/dental_ladder_zeroshot.json`、`data/eval_results_en_dental_weakteacher.json`

## 目录内容

- `data/test_medqa.jsonl`（1273）、`test_medmcqa.jsonl`（4183）、`test_mmlu.jsonl`（2837）、`test_pubmedqa.jsonl`（1000）
- `scripts/eval_ladder.py`：零样本阶梯（4 个模型在英文牙科 980 题的零样本）
- `scripts/eval_weakteacher.py`：弱教师组合评估（训练后学生 vs 弱教师）

## 依赖（重要）

本文件夹是**对 fullEnglish 主实验已训练 adapter 的再分析**，不重复训练：

- 学生训练后 adapter 复用 `fullEnglish/03_main_distill/runs/32B_a00_*`（Qwen2.5-32B）与 `Llama70B_a00_*`（Llama-70B，QLoRA 4bit）的 `best/`。
- 全科训练后数字（75.64% / 76.27%）来自 `fullEnglish/03_main_distill/runs/eval_results.json`。
- 基座模型：`../models/Qwen3-32B`、`../models/Qwen2.5-32B-Instruct`、`../models/Qwen2.5-14B-Instruct`、`../models/Llama-3.3-70B-Instruct`。
- 训练代码：`../shared/train_choice_head_distill.py`。

## 执行步骤

```bash
cd 23_english_weakteacher_surpass

# 1. 测零样本阶梯（4 模型在英文牙科 980 题的零样本）
python scripts/eval_ladder.py

# 2. 弱教师组合评估（Llama-70B 训练后 3-seed 牙科成绩；Qwen2.5-32B 的 65.65% 已知）
python scripts/eval_weakteacher.py
```

## 关键结论

弱教师（Qwen3-32B）领先学生 <4pp → 学生训练后全部反超；强教师（flash）领先 7~14pp → 学生全部追不上。同一规律正反两向成立。
